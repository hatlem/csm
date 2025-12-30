"""
Authentication middleware with API key and JWT support.

Features:
- API key authentication
- JWT token validation
- Scope-based authorization
- Request context enrichment
"""

import hashlib
import time
from typing import Optional, List, Callable, Any
from dataclasses import dataclass
from functools import wraps

from fastapi import Request, Response, Depends, HTTPException
from fastapi.security import APIKeyHeader, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware

from api.core.logging import get_logger
from api.core.errors import AuthenticationError, AuthorizationError

logger = get_logger(__name__)

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_token = HTTPBearer(auto_error=False)


@dataclass
class AuthContext:
    """Authentication context for a request."""

    authenticated: bool = False
    api_key_id: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = []
        if self.metadata is None:
            self.metadata = {}

    def has_scope(self, scope: str) -> bool:
        """Check if context has a specific scope."""
        return scope in self.scopes or "*" in self.scopes


class APIKeyValidator:
    """
    API key validation.

    Keys are stored as hashes in the database/config.
    """

    def __init__(self, keys: Optional[dict[str, dict]] = None):
        """
        Initialize with API keys config.

        keys format:
        {
            "key_hash": {
                "id": "key_id",
                "name": "My API Key",
                "scopes": ["synthesize", "profiles:read"],
                "rate_limit": 100,
            }
        }
        """
        self._keys = keys or {}
        self._simple_key: Optional[str] = None

    def set_simple_key(self, key: str) -> None:
        """Set a simple API key (for basic setups)."""
        self._simple_key = key

    def add_key(
        self,
        key: str,
        key_id: str,
        name: str,
        scopes: List[str],
        rate_limit: int = 100,
    ) -> None:
        """Add an API key."""
        key_hash = self._hash_key(key)
        self._keys[key_hash] = {
            "id": key_id,
            "name": name,
            "scopes": scopes,
            "rate_limit": rate_limit,
        }

    def _hash_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def validate(self, api_key: str) -> Optional[AuthContext]:
        """Validate an API key and return auth context."""
        # Check simple key first
        if self._simple_key and api_key == self._simple_key:
            return AuthContext(
                authenticated=True,
                api_key_id="simple",
                scopes=["*"],  # Full access with simple key
            )

        # Check hashed keys
        key_hash = self._hash_key(api_key)
        if key_hash in self._keys:
            key_info = self._keys[key_hash]
            return AuthContext(
                authenticated=True,
                api_key_id=key_info["id"],
                scopes=key_info["scopes"],
                metadata={"name": key_info["name"]},
            )

        return None


class JWTValidator:
    """
    JWT token validation.

    For integration with external auth providers.
    """

    def __init__(
        self,
        secret: Optional[str] = None,
        algorithm: str = "HS256",
        issuer: Optional[str] = None,
    ):
        self.secret = secret
        self.algorithm = algorithm
        self.issuer = issuer

    def validate(self, token: str) -> Optional[AuthContext]:
        """Validate a JWT token."""
        if not self.secret:
            return None

        try:
            import jwt

            payload = jwt.decode(
                token,
                self.secret,
                algorithms=[self.algorithm],
                issuer=self.issuer,
            )

            return AuthContext(
                authenticated=True,
                user_id=payload.get("sub"),
                scopes=payload.get("scopes", []),
                metadata=payload,
            )
        except ImportError:
            logger.warning("PyJWT not installed, JWT validation disabled")
            return None
        except Exception as e:
            logger.debug(f"JWT validation failed: {e}")
            return None


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware.

    Validates API keys and JWT tokens, adds auth context to request.
    """

    def __init__(
        self,
        app,
        api_key_validator: Optional[APIKeyValidator] = None,
        jwt_validator: Optional[JWTValidator] = None,
        exclude_paths: Optional[List[str]] = None,
        require_auth: bool = True,
    ):
        super().__init__(app)
        self.api_key_validator = api_key_validator or APIKeyValidator()
        self.jwt_validator = jwt_validator
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json",
            "/",
        ]
        self.require_auth = require_auth

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip excluded paths
        if any(request.url.path.startswith(p) for p in self.exclude_paths):
            request.state.auth = AuthContext()
            return await call_next(request)

        auth_context = None

        # Try API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            auth_context = self.api_key_validator.validate(api_key)

        # Try Bearer token
        if not auth_context:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header[7:]
                if self.jwt_validator:
                    auth_context = self.jwt_validator.validate(token)

        # Set auth context
        if auth_context:
            request.state.auth = auth_context
            logger.debug(
                "Request authenticated",
                extra={
                    "api_key_id": auth_context.api_key_id,
                    "user_id": auth_context.user_id,
                    "scopes": auth_context.scopes,
                },
            )
        else:
            request.state.auth = AuthContext()

            if self.require_auth:
                raise AuthenticationError("Valid API key or token required")

        return await call_next(request)


def require_auth(func: Callable) -> Callable:
    """
    Decorator to require authentication.

    Usage:
        @router.get("/protected")
        @require_auth
        async def protected_endpoint(request: Request):
            ...
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Find request in args or kwargs
        request = kwargs.get("request")
        if not request:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

        if not request:
            raise ValueError("Request not found in function arguments")

        auth: AuthContext = getattr(request.state, "auth", None)
        if not auth or not auth.authenticated:
            raise AuthenticationError()

        return await func(*args, **kwargs)

    return wrapper


def require_scopes(*required_scopes: str) -> Callable:
    """
    Decorator to require specific scopes.

    Usage:
        @router.post("/train")
        @require_scopes("training:write")
        async def start_training(request: Request):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if not request:
                raise ValueError("Request not found in function arguments")

            auth: AuthContext = getattr(request.state, "auth", None)
            if not auth or not auth.authenticated:
                raise AuthenticationError()

            for scope in required_scopes:
                if not auth.has_scope(scope):
                    raise AuthorizationError(
                        f"Missing required scope: {scope}"
                    )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def get_auth_context(request: Request) -> AuthContext:
    """FastAPI dependency to get auth context."""
    return getattr(request.state, "auth", AuthContext())


async def get_current_api_key(
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[str]:
    """FastAPI dependency to get API key."""
    return api_key
