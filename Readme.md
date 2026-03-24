# Aina Main Backend

Public-facing FastAPI service that sits between the frontend (HTML/Alpine.js) and the internal GPU worker (`hmr_worker.py`).

## Architecture

```
Browser (demo.html)
    │
    │  HTTPS  (CORS-gated)
    ▼
Aina Main Backend  ← this service, port 8000
    │   Supabase Postgres  — session rows
    │   Supabase Storage   — face & body photos
    │
    │  HTTP  (internal network only)
    ▼
GPU Worker (hmr_worker.py)  port 8002
    └─ HMR2 + SMPL mesh measurements
```

**The GPU worker is never called by the frontend.** All traffic goes through this backend.

---

## Quickstart

### 1. Prerequisites

- Python 3.11+
- A Supabase project with the migration applied (see below)

### 2. Clone & install

```bash
cd aina-backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment

```bash
cp .env.example .env
# Edit .env — fill in SUPABASE_URL, SUPABASE_SERVICE_KEY, WORKER_URL
```

**Key variables:**

| Variable | Description |
|---|---|
| `SUPABASE_URL` | Your project URL from Supabase Dashboard → Settings → API |
| `SUPABASE_SECRET_KEY` | **Service role** key (not anon key) — Dashboard → Settings → API |
| `WORKER_URL` | Internal URL of the GPU worker, e.g. `http://10.0.0.5:8002` |
| `ALLOWED_ORIGINS` | Comma-separated frontend origins, e.g. `https://yourstore.com` |
| `SESSION_COOKIE_SECURE` | Set `false` for local HTTP dev, `true` in production |

### 4. Run the Supabase migration

Open Supabase Dashboard → SQL Editor → New query, paste `supabase_migration.sql`, and run it.

This creates:
- `public.sessions` table with RLS enabled
- `aina-face-photos` and `aina-body-photos` storage buckets (private)
- All indexes and triggers

### 5. Start the server

```bash
# Development (auto-reload)
python main.py

# Or with uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs` (disabled in production).

---

## API Reference

All routes use a **session cookie** (`aina_session`) for state. The cookie is `HttpOnly` (JS can't read it) and set on the first `POST /session` call.

### Flow overview

```
POST /session              ← Step 2: gender + height + weight confirmed → sets cookie
POST /session/face-photo   ← Step 3: upload face photo (optional)
POST /session/extract      ← Step 4: upload body photo + trigger AI extraction (single call)
GET  /session              ← Read current session at any point
POST /session/reset        ← Clear session cookie (start over)
```

### `POST /session`

Called **once** when the user confirms gender + height + weight at Step 2. Creates the session and sets the `HttpOnly` cookie for all subsequent requests. If a valid cookie already exists (e.g. user went back and changed a value), patches the existing session instead of creating a duplicate.

**Body (JSON):**
```json
{
  "gender": "male",
  "height_cm": 175.0,
  "weight_kg": 70.0
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "status": "initiated",
  "gender": "male",
  "height_cm": 175.0,
  "weight_kg": 70.0,
  "has_face_photo": false,
  "has_body_photo": false,
  "measurements": null,
  "error_message": null
}
```

Sets cookie: `aina_session=<uuid>; HttpOnly; SameSite=Lax`

---

### `POST /session/face-photo`

**Content-Type:** `multipart/form-data`
**Field:** `file` (image/jpeg, image/png, image/webp, max 10MB)
**Cookie:** `aina_session` required

**Response:**
```json
{
  "session_id": "uuid",
  "face_photo_url": "https://...supabase.co/...?token=...",
  "status": "images_uploaded"
}
```

---

### `POST /session/extract`

Single endpoint for Step 4. Accepts the body photo, stores it in Supabase Storage, then immediately forwards it to the GPU worker for AI measurement extraction. No separate body-photo upload step.

**Content-Type:** `multipart/form-data`
**Field:** `file` (image/jpeg, image/png, image/webp, max 10MB)
**Cookie:** `aina_session` required

**Response (200):**
```json
{
  "session_id": "uuid",
  "status": "completed",
  "measurements": {
    "chest_cm": 95.2,
    "waist_cm": 82.1,
    "hip_cm": 98.4,
    "shoulder_width_cm": 42.3,
    "bmi": 22.5
  }
}
```

**Error responses:**

| HTTP | error_code | Cause |
|---|---|---|
| 409 | `ALREADY_PROCESSING` | Concurrent extract call on same session |
| 422 | `INVALID_FILE_TYPE` | Not a JPG/PNG/WebP |
| 422 | `FILE_TOO_LARGE` | Over 10MB |
| 422 | `NO_PERSON_DETECTED` | Worker couldn't find a person in the image |
| 422 | `BODY_PARTS_OCCLUDED` | Full body not visible |
| 422 | `LANDSCAPE_IMAGE` | Image is horizontal |
| 503 | `WORKER_UNAVAILABLE` | GPU worker is down |
| 504 | `WORKER_TIMEOUT` | Worker took too long |

---

## Project structure

```
aina-backend/
├── main.py                    # FastAPI app, all routes
├── config.py                  # Settings from .env (pydantic-settings)
├── supabase_migration.sql     # Run once in Supabase SQL Editor
├── requirements.txt
├── .env.example
├── db/
│   └── supabase_client.py     # Cached Supabase client
├── models/
│   └── schemas.py             # All Pydantic request/response models
├── services/
│   ├── session_service.py     # sessions table CRUD
│   ├── storage_service.py     # Supabase Storage upload/download/sign
│   └── worker_service.py      # Internal HTTP client → GPU worker
└── logs/
    └── app.log                # Rotating log file
```

---

## Updating `demo.html`

Replace the direct worker calls in `demo.html` with these backend endpoints:

| Step | Old (direct to worker) | New (via backend) |
|---|---|---|
| Step 2 | — | `POST /session` with gender + height + weight |
| Step 3 | — | `POST /session/face-photo` (optional) |
| Step 4 | `POST /extract_3d` with multipart | `POST /session/extract` with body photo file |

The frontend no longer needs the worker URL or the ngrok configurator at all.

---

## Production checklist

- [ ] Set `SESSION_COOKIE_SECURE=true`
- [ ] Set `APP_ENV=production` (disables `/docs`)
- [ ] Restrict `ALLOWED_ORIGINS` to your actual domain(s)
- [ ] Use a reverse proxy (nginx/Caddy) in front of uvicorn
- [ ] Ensure `WORKER_URL` is a private/internal network address
- [ ] Rotate `SUPABASE_SERVICE_KEY` if it was ever committed to git
- [ ] Set up a cron job to purge stale `initiated`/`failed` sessions older than 48h