"""
Redis-based job queue for long-running tasks.

Features:
- Persistent job storage
- Priority queues
- Job status tracking
- Dead letter queue for failed jobs
- Automatic retry with backoff
"""

import json
import asyncio
import uuid
import time
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum

from api.core.logging import get_logger
from api.core.metrics import get_metrics

logger = get_logger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD = "dead"  # Max retries exceeded


@dataclass
class Job:
    """A queued job."""

    id: str
    queue: str
    payload: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    priority: int = 0  # Higher = more priority
    attempts: int = 0
    max_attempts: int = 3
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        data["status"] = JobStatus(data["status"])
        return cls(**data)


class JobQueue:
    """
    Redis-based job queue.

    Usage:
        queue = JobQueue(redis_url="redis://localhost:6379")
        await queue.initialize()

        # Enqueue a job
        job_id = await queue.enqueue("training", {"profile_id": "123"})

        # Process jobs
        async def worker():
            while True:
                job = await queue.dequeue("training")
                if job:
                    try:
                        result = await process_job(job)
                        await queue.complete(job.id, result)
                    except Exception as e:
                        await queue.fail(job.id, str(e))
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        prefix: str = "csm:queue:",
    ):
        self.redis_url = redis_url
        self.prefix = prefix
        self._redis = None
        self._workers: List[asyncio.Task] = []
        self._handlers: Dict[str, Callable] = {}
        self._shutdown = False
        self._metrics = get_metrics()

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            logger.info(f"Connected to Redis: {self.redis_url}")
        except ImportError:
            logger.warning("redis package not installed, using in-memory queue")
            self._redis = None
            self._memory_queues: Dict[str, List[Job]] = {}
            self._memory_jobs: Dict[str, Job] = {}
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis = None
            self._memory_queues: Dict[str, List[Job]] = {}
            self._memory_jobs: Dict[str, Job] = {}

    async def shutdown(self) -> None:
        """Shutdown the queue."""
        self._shutdown = True

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        # Close Redis connection
        if self._redis:
            await self._redis.close()

    def _queue_key(self, queue: str) -> str:
        return f"{self.prefix}{queue}"

    def _job_key(self, job_id: str) -> str:
        return f"{self.prefix}job:{job_id}"

    def _processing_key(self, queue: str) -> str:
        return f"{self.prefix}{queue}:processing"

    def _dead_letter_key(self, queue: str) -> str:
        return f"{self.prefix}{queue}:dead"

    async def enqueue(
        self,
        queue: str,
        payload: Dict[str, Any],
        priority: int = 0,
        max_attempts: int = 3,
    ) -> str:
        """
        Add a job to the queue.

        Args:
            queue: Queue name
            payload: Job data
            priority: Priority (higher = more important)
            max_attempts: Maximum retry attempts

        Returns:
            Job ID
        """
        job = Job(
            id=str(uuid.uuid4()),
            queue=queue,
            payload=payload,
            priority=priority,
            max_attempts=max_attempts,
        )

        if self._redis:
            # Store job data
            await self._redis.set(
                self._job_key(job.id),
                json.dumps(job.to_dict()),
                ex=86400 * 7,  # 7 day TTL
            )

            # Add to queue (sorted by priority)
            await self._redis.zadd(
                self._queue_key(queue),
                {job.id: priority},
            )
        else:
            # In-memory fallback
            if queue not in self._memory_queues:
                self._memory_queues[queue] = []
            self._memory_queues[queue].append(job)
            self._memory_queues[queue].sort(key=lambda j: j.priority, reverse=True)
            self._memory_jobs[job.id] = job

        self._metrics.set_queue_size(queue, await self.queue_length(queue))

        logger.info(
            f"Enqueued job {job.id} to {queue}",
            extra={"job_id": job.id, "queue": queue, "priority": priority},
        )

        return job.id

    async def dequeue(
        self,
        queue: str,
        timeout: float = 0,
    ) -> Optional[Job]:
        """
        Get the next job from the queue.

        Args:
            queue: Queue name
            timeout: How long to wait for a job (0 = no wait)

        Returns:
            Job or None
        """
        if self._redis:
            # Get highest priority job
            result = await self._redis.zpopmax(self._queue_key(queue))

            if not result:
                if timeout > 0:
                    # Wait for new jobs
                    await asyncio.sleep(min(timeout, 1))
                return None

            job_id = result[0][0]

            # Get job data
            job_data = await self._redis.get(self._job_key(job_id))
            if not job_data:
                return None

            job = Job.from_dict(json.loads(job_data))

            # Update status
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow().isoformat()
            job.attempts += 1

            await self._redis.set(
                self._job_key(job.id),
                json.dumps(job.to_dict()),
            )

            # Add to processing set
            await self._redis.sadd(self._processing_key(queue), job.id)

        else:
            # In-memory fallback
            if queue not in self._memory_queues or not self._memory_queues[queue]:
                if timeout > 0:
                    await asyncio.sleep(min(timeout, 1))
                return None

            job = self._memory_queues[queue].pop(0)
            job.status = JobStatus.PROCESSING
            job.started_at = datetime.utcnow().isoformat()
            job.attempts += 1

        self._metrics.set_queue_size(queue, await self.queue_length(queue))
        return job

    async def complete(
        self,
        job_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mark a job as completed."""
        if self._redis:
            job_data = await self._redis.get(self._job_key(job_id))
            if not job_data:
                return

            job = Job.from_dict(json.loads(job_data))
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow().isoformat()
            job.result = result

            await self._redis.set(
                self._job_key(job.id),
                json.dumps(job.to_dict()),
                ex=86400,  # Keep for 1 day after completion
            )

            # Remove from processing set
            await self._redis.srem(self._processing_key(job.queue), job.id)

        else:
            if job_id in self._memory_jobs:
                job = self._memory_jobs[job_id]
                job.status = JobStatus.COMPLETED
                job.completed_at = datetime.utcnow().isoformat()
                job.result = result

        logger.info(f"Job {job_id} completed")

    async def fail(
        self,
        job_id: str,
        error: str,
        retry: bool = True,
    ) -> None:
        """Mark a job as failed, optionally retry."""
        if self._redis:
            job_data = await self._redis.get(self._job_key(job_id))
            if not job_data:
                return

            job = Job.from_dict(json.loads(job_data))
        else:
            job = self._memory_jobs.get(job_id)
            if not job:
                return

        job.error = error

        # Check if we should retry
        if retry and job.attempts < job.max_attempts:
            job.status = JobStatus.RETRYING

            # Calculate backoff delay
            delay = min(30 * (2 ** (job.attempts - 1)), 300)  # Max 5 min

            logger.warning(
                f"Job {job_id} failed, retrying in {delay}s (attempt {job.attempts}/{job.max_attempts})",
                extra={"job_id": job_id, "error": error, "attempt": job.attempts},
            )

            # Re-enqueue with delay (simplified - just re-add)
            await asyncio.sleep(delay)
            await self.enqueue(
                job.queue,
                job.payload,
                priority=job.priority,
                max_attempts=job.max_attempts - job.attempts,
            )
        else:
            # Move to dead letter queue
            job.status = JobStatus.DEAD
            job.completed_at = datetime.utcnow().isoformat()

            if self._redis:
                await self._redis.lpush(
                    self._dead_letter_key(job.queue),
                    json.dumps(job.to_dict()),
                )

            logger.error(
                f"Job {job_id} permanently failed",
                extra={"job_id": job_id, "error": error},
            )

        # Update job
        if self._redis:
            await self._redis.set(
                self._job_key(job.id),
                json.dumps(job.to_dict()),
            )
            await self._redis.srem(self._processing_key(job.queue), job.id)

    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        if self._redis:
            job_data = await self._redis.get(self._job_key(job_id))
            if job_data:
                return Job.from_dict(json.loads(job_data))
        else:
            return self._memory_jobs.get(job_id)
        return None

    async def queue_length(self, queue: str) -> int:
        """Get queue length."""
        if self._redis:
            return await self._redis.zcard(self._queue_key(queue))
        else:
            return len(self._memory_queues.get(queue, []))

    def register_handler(
        self,
        queue: str,
        handler: Callable[[Job], Any],
    ) -> None:
        """Register a handler for a queue."""
        self._handlers[queue] = handler

    async def start_worker(
        self,
        queue: str,
        concurrency: int = 1,
    ) -> None:
        """Start workers for a queue."""
        for i in range(concurrency):
            task = asyncio.create_task(self._worker_loop(queue, i))
            self._workers.append(task)
            logger.info(f"Started worker {i} for queue {queue}")

    async def _worker_loop(self, queue: str, worker_id: int) -> None:
        """Worker loop that processes jobs."""
        handler = self._handlers.get(queue)
        if not handler:
            logger.error(f"No handler registered for queue {queue}")
            return

        while not self._shutdown:
            try:
                job = await self.dequeue(queue, timeout=5)
                if not job:
                    continue

                start_time = time.time()
                try:
                    result = await handler(job)
                    await self.complete(job.id, result)

                    self._metrics.record_queue_processing(
                        queue, time.time() - start_time
                    )
                except Exception as e:
                    await self.fail(job.id, str(e))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
