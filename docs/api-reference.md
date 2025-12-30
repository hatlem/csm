# CSM Voice API Reference

## Base URL

```
https://1s8c803ve11jp3-8000.proxy.runpod.net
```

## HTTP Endpoints

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

**Status Codes:**
- `200`: Service is healthy
- `503`: Model not loaded

---

### POST /synthesize

Synthesize speech from text.

**Request:**
```json
{
  "text": "Hello, how are you?",
  "speaker": 0,
  "max_audio_length_ms": 30000,
  "output_format": "wav"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | *required* | Text to synthesize |
| `speaker` | int | 0 | Speaker ID (0-based) |
| `max_audio_length_ms` | int | 30000 | Max audio length in ms |
| `output_format` | string | "wav" | "wav" or "mulaw" |

**Response:**
```json
{
  "audio_base64": "UklGRk...",
  "sample_rate": 24000,
  "duration_ms": 1500,
  "generation_time_ms": 800,
  "format": "wav"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `audio_base64` | string | Base64-encoded audio |
| `sample_rate` | int | Sample rate (24000 for wav, 8000 for mulaw) |
| `duration_ms` | int | Audio duration in milliseconds |
| `generation_time_ms` | int | Time taken to generate |
| `format` | string | "wav" or "mulaw_8khz" |

**Example:**
```bash
curl -X POST https://1s8c803ve11jp3-8000.proxy.runpod.net/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "max_audio_length_ms": 5000}'
```

---

## WebSocket Endpoint

### /v1/native/ws/{session_id}

Real-time bidirectional voice synthesis for Human-Like integration.

**Connection:**
```javascript
const ws = new WebSocket("wss://1s8c803ve11jp3-8000.proxy.runpod.net/v1/native/ws/my-session-123");
```

**URL Parameters:**
| Parameter | Description |
|-----------|-------------|
| `session_id` | Unique session identifier (use call SID or UUID) |

---

### Client → Server Messages

#### Synthesize Request
```json
{
  "type": "synthesize",
  "text": "Hello, how are you?",
  "voice_profile_id": "default"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | Must be "synthesize" |
| `text` | string | Text to synthesize |
| `voice_profile_id` | string | Voice profile ID (for future multi-voice support) |

#### Ping (Keepalive)
```json
{
  "type": "ping"
}
```

#### Close Connection
```json
{
  "type": "close"
}
```

---

### Server → Client Messages

#### Audio Response
```json
{
  "type": "audio",
  "audio_base64": "f3+AgICAgH9/...",
  "duration_ms": 1500,
  "generation_time_ms": 800,
  "format": "mulaw_8khz"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | "audio" |
| `audio_base64` | string | Base64-encoded mulaw audio |
| `duration_ms` | int | Audio duration in milliseconds |
| `generation_time_ms` | int | Time taken to generate |
| `format` | string | Always "mulaw_8khz" for WebSocket |

**Audio Format:**
- Encoding: G.711 mu-law
- Sample Rate: 8,000 Hz
- Bit Depth: 8-bit (uint8)
- Ready for Twilio Media Streams

#### Pong Response
```json
{
  "type": "pong"
}
```

#### Error Response
```json
{
  "type": "error",
  "error": "Model not loaded"
}
```

---

## WebSocket Example (JavaScript)

```javascript
class CSMClient {
  constructor(sessionId) {
    this.sessionId = sessionId;
    this.ws = null;
  }

  connect() {
    return new Promise((resolve, reject) => {
      const url = `wss://1s8c803ve11jp3-8000.proxy.runpod.net/v1/native/ws/${this.sessionId}`;
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log("Connected to CSM");
        resolve();
      };

      this.ws.onmessage = (event) => {
        const message = JSON.parse(event.data);

        if (message.type === "audio") {
          // Forward to Twilio
          this.onAudio(message.audio_base64, message.duration_ms);
        } else if (message.type === "pong") {
          console.log("Keepalive OK");
        } else if (message.type === "error") {
          console.error("CSM error:", message.error);
        }
      };

      this.ws.onerror = (error) => {
        reject(error);
      };

      this.ws.onclose = () => {
        console.log("Disconnected from CSM");
      };
    });
  }

  synthesize(text, voiceProfileId = "default") {
    this.ws.send(JSON.stringify({
      type: "synthesize",
      text: text,
      voice_profile_id: voiceProfileId,
    }));
  }

  ping() {
    this.ws.send(JSON.stringify({ type: "ping" }));
  }

  close() {
    this.ws.send(JSON.stringify({ type: "close" }));
    this.ws.close();
  }

  // Override this to handle audio
  onAudio(base64Audio, durationMs) {
    console.log(`Received ${durationMs}ms of audio`);
  }
}

// Usage
const client = new CSMClient("call-123");
await client.connect();
client.synthesize("Hello, how can I help you today?");
```

---

## WebSocket Example (Python)

```python
import asyncio
import websockets
import json

async def csm_client(session_id: str, text: str):
    url = f"wss://1s8c803ve11jp3-8000.proxy.runpod.net/v1/native/ws/{session_id}"

    async with websockets.connect(url) as ws:
        # Send synthesis request
        await ws.send(json.dumps({
            "type": "synthesize",
            "text": text,
            "voice_profile_id": "default"
        }))

        # Wait for audio response
        response = await ws.recv()
        message = json.loads(response)

        if message["type"] == "audio":
            print(f"Generated {message['duration_ms']}ms of audio")
            return message["audio_base64"]
        elif message["type"] == "error":
            raise Exception(message["error"])

# Usage
audio = asyncio.run(csm_client("session-123", "Hello world"))
```

---

## Error Handling

### HTTP Errors

| Status | Error | Description |
|--------|-------|-------------|
| 400 | Bad Request | Invalid JSON or missing required fields |
| 422 | Validation Error | Field validation failed |
| 503 | Service Unavailable | Model not loaded |
| 500 | Internal Server Error | Synthesis failed |

### WebSocket Errors

Errors are returned as JSON messages with `type: "error"`:

```json
{
  "type": "error",
  "error": "Description of the error"
}
```

Common errors:
- "Model not loaded" - Wait for startup or check /health
- "Invalid message format" - Check JSON structure
- "Empty text" - Provide non-empty text

---

## Rate Limits

Currently no rate limits enforced. For production:
- Consider adding authentication
- Implement per-session rate limiting
- Monitor GPU utilization

---

## Performance Tips

1. **Use WebSocket for streaming**: Lower latency than HTTP
2. **Keep connections alive**: Send pings every 30s
3. **Batch short utterances**: Combine small texts to reduce overhead
4. **Set appropriate max_audio_length_ms**: Don't request more than needed
5. **Use mulaw format**: Smaller than WAV, ready for Twilio
