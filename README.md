# Face-ViT-Arc

A serverless face recognition API deployed on [Modal](https://modal.com). Supports 1:1 face verification, 1:N face identification, face registration, and audit logging. The ML inference runs on a GPU container using a ViT-Base DINOv2 backbone; the HTTP API runs on a separate CPU container.

---

## Project Structure

```
Face-ViT-Arc/
    main.py              Modal entry point
    requirements.txt     Local dependencies
    app/
        __init__.py
        config.py        Modal infrastructure, paths, and inference constants
        models.py        Pydantic request and response schemas
        auth.py          API key authentication dependency
        db.py            Database class (SQLite - identities + audit logs)
        engine.py        FaceEngine GPU class (embedding, verify, identify)
        server.py        FastAPI application factory and route definitions
```

---

## Architecture

The app is split into two Modal components that run independently:

### FaceEngine (GPU - NVIDIA T4)

Defined in `app/engine.py`. Decorated with `@app.cls(gpu="T4")`, this is a stateful Modal Class that holds models in GPU memory across requests.

**Startup (`@modal.enter`)**

- Loads MTCNN (facenet-pytorch) as the face detector
- Loads `vit_base_patch14_dinov2.lvd142m` from timm as the embedding model (768-dim output)
- Optionally loads fine-tuned weights from `/data/custom_model.pth` on the volume
- Loads or initialises the FAISS `IndexIDMap(IndexFlatIP(768))` from `/data/faces.index`

**Embedding pipeline (`_embed`)**

1. Decode raw image bytes to a PIL image
2. Run MTCNN to detect and crop the face
3. Apply the ViT preprocessing transform
4. Run the ViT model to produce a 768-dim float32 vector
5. L2-normalise the vector (makes dot product equivalent to cosine similarity)

**Remote methods**

| Method     | Arguments                        | Returns                | Description                                       |
| ---------- | -------------------------------- | ---------------------- | ------------------------------------------------- |
| `register` | `image_bytes`, `faiss_idx`       | `bool`                 | Adds an embedding at the given index ID           |
| `verify`   | `image_bytes_a`, `image_bytes_b` | `float \| None`        | Returns cosine similarity score between two faces |
| `identify` | `image_bytes`                    | `(int, float) \| None` | Returns the nearest `faiss_idx` and its score     |
| `remove`   | `faiss_idx`                      | `bool`                 | Removes an embedding from the index               |
| `count`    | -                                | `int`                  | Returns total registered faces                    |

### fastapi_app (CPU - always warm)

Defined in `app/server.py`, served via `main.py`. Decorated with `@app.function(min_containers=1)` and `@modal.asgi_app()`. Stays warm at all times to avoid cold starts on HTTP requests.

---

## Persistent Storage

All persistent data lives on a Modal Volume (`face-pro-storage`) mounted at `/data` inside both containers.

| File                     | Contents                                            |
| ------------------------ | --------------------------------------------------- |
| `/data/faces.index`      | FAISS index storing all face embeddings             |
| `/data/logs.db`          | SQLite database with `identities` and `logs` tables |
| `/data/custom_model.pth` | Optional fine-tuned ViT weights (loaded if present) |

### SQLite Schema

```sql
CREATE TABLE identities (
    faiss_idx    INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    registered_at TEXT   NOT NULL
);

CREATE TABLE logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    operation   TEXT    NOT NULL,
    result      TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    feedback    TEXT,
    created_at  TEXT    NOT NULL
);
```

---

## API Endpoints

All endpoints except `/api/health` require the `X-API-Key` header.

### `GET /api/health`

Public. Returns server status and total registered faces.

**Response**

```json
{ "status": "ok", "registered_faces": 42 }
```

---

### `POST /api/register`

Register a new face in the database.

**Form data**

- `file` - image file (JPEG, PNG, or WebP, max 10 MB)
- `name` - string name to associate with the face

**Response**

```json
{ "success": true, "faiss_idx": 0, "message": "Registered 'Alice' at index 0" }
```

---

### `POST /api/verify`

1:1 verification. Checks whether two images show the same person.

**Form data**

- `file1` - first image
- `file2` - second image

**Response**

```json
{ "match": true, "confidence": 0.87, "log_id": 12 }
```

Match threshold: `0.6` (cosine similarity).

---

### `POST /api/identify`

1:N identification. Finds the closest registered identity to a probe image.

**Form data**

- `file` - probe image

**Response**

```json
{ "faiss_idx": 3, "name": "Alice", "confidence": 0.91, "log_id": 13 }
```

---

### `GET /api/history`

Returns the 50 most recent log entries in descending order.

**Response**

```json
[
  {
    "id": 13,
    "operation": "identify",
    "result": "Alice",
    "confidence": 0.91,
    "feedback": null,
    "created_at": "2026-02-22T10:45:00+00:00"
  }
]
```

---

### `POST /api/feedback`

Attach a feedback label to an existing log entry.

**JSON body**

```json
{ "log_id": 13, "feedback": "correct" }
```

**Response**

```json
{ "status": "ok" }
```

---

## Error Responses

| Status                         | Condition                              |
| ------------------------------ | -------------------------------------- |
| `401 Unauthorized`             | Missing or invalid `X-API-Key`         |
| `404 Not Found`                | Log entry or identity does not exist   |
| `413 Request Entity Too Large` | Image exceeds 10 MB                    |
| `415 Unsupported Media Type`   | File is not JPEG, PNG, or WebP         |
| `422 Unprocessable Entity`     | No face detected in the uploaded image |

---

## Setup

### 1. Install local dependencies

```bash
pip install -r requirements.txt
```

### 2. Authenticate with Modal

```bash
modal setup
```

### 3. Create the API key secret

```bash
modal secret create face-api-secret API_KEY=your-key-here
```

For local development, create a separate secret that bypasses auth:

```bash
modal secret create face-api-secret-dev API_KEY=dev API_KEY=dev DEV_MODE=true
```

---

## Running

### Development (live reload)

```bash
modal serve main.py
```

Modal starts the app and prints a temporary HTTPS URL. Changes to any file in `app/` are picked up immediately without redeployment. Navigate to `<url>/docs` to use the interactive FastAPI docs UI.

To use the dev secret instead of the production one, temporarily edit `main.py`:

```python
secrets=[modal.Secret.from_name("face-api-secret-dev")],
```

### Production deployment

```bash
modal deploy main.py
```

Assigns a permanent URL of the form:

```
https://<workspace>--face-recognition-suite-fastapi-app.modal.run
```

---

## Example curl Calls

```bash
BASE=https://<your-modal-url>
KEY=your-key-here

# Health (no auth)
curl $BASE/api/health

# Register
curl -X POST $BASE/api/register \
  -H "X-API-Key: $KEY" \
  -F "file=@alice.jpg" \
  -F "name=Alice"

# Verify
curl -X POST $BASE/api/verify \
  -H "X-API-Key: $KEY" \
  -F "file1=@alice1.jpg" \
  -F "file2=@alice2.jpg"

# Identify
curl -X POST $BASE/api/identify \
  -H "X-API-Key: $KEY" \
  -F "file=@probe.jpg"

# History
curl $BASE/api/history \
  -H "X-API-Key: $KEY"

# Feedback
curl -X POST $BASE/api/feedback \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"log_id": 13, "feedback": "correct"}'
```

Endpoint Link: [https://sayangupta840--face-recognition-suite-fastapi-app.modal.run/](https://sayangupta840--face-recognition-suite-fastapi-app.modal.run/)
