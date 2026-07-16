# AI Gen API v2 - Music Generator API

Base URL:

```text
https://YOUR_POD_ID-8001.proxy.runpod.net
```

The music worker uses ACE-Step's upstream async REST API. A request creates a task, polling returns the audio URL and lyrics.

## Authentication

If `ACESTEP_API_KEY` is set on the pod, send:

```http
Authorization: Bearer YOUR_KEY
```

If no key is set, the endpoints are unauthenticated. Use a key before production.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Health check |
| `POST` | `/release_task` | Submit music generation task |
| `POST` | `/query_result` | Poll one or more task IDs |
| `GET` | `/v1/audio?path=...` | Download generated audio |
| `GET` | `/v1/models` | List loaded/available models |
| `GET` | `/v1/stats` | Queue/runtime stats |
| `POST` | `/format_input` | Format/enhance prompt + lyrics with LM |
| `POST` | `/create_random_sample` | Generate random prompt/lyrics metadata |

Interactive docs:

```text
https://YOUR_POD_ID-8001.proxy.runpod.net/docs
```

## Generate Music

Submit a task:

```bash
curl -X POST https://YOUR_POD_ID-8001.proxy.runpod.net/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "upbeat pop jingle, bright synths, clean vocal, radio ready",
    "lyrics": "[Verse]\nWe build the spark\nWe light the way\n[Chorus]\nCreate the sound\nAnd press play",
    "audio_duration": 10,
    "inference_steps": 8,
    "batch_size": 1,
    "audio_format": "mp3",
    "thinking": false
  }'
```

Response:

```json
{
  "data": {
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "queued",
    "queue_position": 1
  },
  "code": 200,
  "error": null
}
```

## Poll Result

```bash
curl -X POST https://YOUR_POD_ID-8001.proxy.runpod.net/query_result \
  -H "Content-Type: application/json" \
  -d '{"task_id_list": ["550e8400-e29b-41d4-a716-446655440000"]}'
```

Status codes:

| Status | Meaning |
|--------|---------|
| `0` | Queued/running |
| `1` | Succeeded |
| `2` | Failed |

Successful response shape:

```json
{
  "data": [
    {
      "task_id": "550e8400-e29b-41d4-a716-446655440000",
      "status": 1,
      "result": "[{\"file\":\"/v1/audio?path=...\",\"lyrics\":\"[Verse]...\",\"metas\":{\"lyrics\":\"[Verse]...\",\"duration\":10}}]"
    }
  ],
  "code": 200,
  "error": null
}
```

Important: `result` is a JSON string. Parse it once before reading fields.

JavaScript example:

```js
const pollResponse = await fetch(`${baseUrl}/query_result`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ task_id_list: [taskId] })
}).then((res) => res.json());

const row = pollResponse.data[0];
const items = JSON.parse(row.result);
const music = items[0];

console.log(music.file);
console.log(music.lyrics);
```

## Lyrics Display

Use these fields after parsing `data[0].result`:

| Field | Use |
|-------|-----|
| `music.lyrics` | Display lyrics used by the final generation |
| `music.metas.lyrics` | Original user-provided lyrics |
| `music.prompt` | Final prompt/caption |
| `music.metas.duration` | Returned duration metadata |

For user-provided lyrics, send `lyrics` in `/release_task`.

For generated lyrics, use description/sample mode:

```bash
curl -X POST https://YOUR_POD_ID-8001.proxy.runpod.net/release_task \
  -H "Content-Type: application/json" \
  -d '{
    "sample_mode": true,
    "sample_query": "a cheerful pop song for a basketball championship intro",
    "audio_duration": 30,
    "thinking": true,
    "batch_size": 1,
    "audio_format": "mp3"
  }'
```

When the task succeeds, display `music.lyrics`.

## Download Audio

The `file` field is usually a relative URL:

```bash
curl "https://YOUR_POD_ID-8001.proxy.runpod.net/v1/audio?path=ENCODED_PATH" \
  -o generated-song.mp3
```

## Useful Parameters

| Field | Default | Notes |
|-------|---------|-------|
| `prompt` | `""` | Music description |
| `lyrics` | `""` | Lyrics to sing; use `[Instrumental]` for instrumental |
| `sample_mode` | `false` | Auto-generate prompt/lyrics from `sample_query` |
| `sample_query` | `""` | Natural-language song idea |
| `thinking` | `false` | Use LM planning; slower but often better |
| `audio_duration` | model default | 10-600 seconds |
| `inference_steps` | `8` | Turbo default; lower is faster |
| `batch_size` | `2` | Use `1` for fastest single result |
| `audio_format` | `mp3` | `mp3`, `wav`, `flac`, `opus`, `aac` |
| `model` | default | Example: `acestep-v15-turbo` |

## Minimal End-to-End Script

```bash
BASE=https://YOUR_POD_ID-8001.proxy.runpod.net

TASK_ID=$(curl -s -X POST "$BASE/release_task" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt":"upbeat pop jingle",
    "lyrics":"[Verse]\nBuild it fast\nMake it bright",
    "audio_duration":10,
    "batch_size":1,
    "audio_format":"mp3",
    "thinking":false
  }' | jq -r '.data.task_id')

while true; do
  RESPONSE=$(curl -s -X POST "$BASE/query_result" \
    -H "Content-Type: application/json" \
    -d "{\"task_id_list\":[\"$TASK_ID\"]}")

  STATUS=$(echo "$RESPONSE" | jq -r '.data[0].status')
  echo "status=$STATUS"

  if [ "$STATUS" = "1" ]; then
    RESULT=$(echo "$RESPONSE" | jq -r '.data[0].result')
    FILE=$(echo "$RESULT" | jq -r '.[0].file')
    LYRICS=$(echo "$RESULT" | jq -r '.[0].lyrics')
    echo "$LYRICS"
    curl "$BASE$FILE" -o generated-song.mp3
    break
  fi

  if [ "$STATUS" = "2" ]; then
    echo "$RESPONSE"
    exit 1
  fi

  sleep 3
done
```
