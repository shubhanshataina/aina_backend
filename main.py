# main.py — Aina Main Backend
# Public-facing FastAPI service.
# Handles CORS, session cookies, image uploads, and worker orchestration.
# The GPU worker (hmr_worker.py) is internal — never called by the frontend directly.

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler

import os
import uvicorn
from fastapi import (
    FastAPI, File, Form, UploadFile,
    Request, Response, HTTPException, Depends,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from models.schemas import (
    Gender, SessionStatus, AvatarStatus,
    SessionCreateRequest, SessionResponse,
    UploadFaceResponse,
    ExtractResponse, Measurements,
    MeasurementUpdateRequest, MeasurementUpdateResponse,
    AvatarResponse,
)
import services.session_service    as session_svc
import services.storage_service    as storage_svc
import services.worker_service     as worker_svc
import services.avatar_service     as avatar_svc
import services.correction_service as correction_svc


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def _setup_logger(name: str, filename: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = RotatingFileHandler(
        os.path.join(LOG_DIR, filename),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger

log = _setup_logger("app", "app.log")


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    log.info("=" * 60)
    log.info("Aina Main Backend starting")
    log.info(f"Environment : {settings.app_env}")
    log.info(f"Worker URL  : {settings.worker_url}")
    log.info(f"Supabase    : {settings.supabase_url}")
    log.info(f"CORS origins ({len(settings.allowed_origins_list)}):")
    for origin in settings.allowed_origins_list:
        log.info(f"  → {origin}")
    log.info(f"Cookie secure  : {settings.session_cookie_secure}")
    log.info(f"Cookie samesite: {settings.session_cookie_samesite}")
    log.info(f"Avatar provider: {settings.avatar_provider}")
    log.info("=" * 60)

    # Warm up the Vertex AI SDK and gRPC channel before accepting requests.
    # This eliminates the first-request timeout caused by lazy SDK init + cold
    # gRPC channel establishment (10-20s) combining with generate_content (15-45s)
    # to exceed the 90s client timeout. See avatar_service.warmup_vertex_sdk().
    await avatar_svc.warmup_vertex_sdk()

    yield
    log.info("Aina Main Backend shutting down")


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title       = "Aina API",
    description = "Main backend for the Aina virtual try-on platform.",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs" if not settings.is_production else None,
    redoc_url   = None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = settings.allowed_origins_list,
    allow_credentials = True,    # required for HttpOnly session cookie
    allow_methods     = ["GET", "POST", "PATCH", "DELETE"],
    # Cannot use ["*"] with allow_credentials=True — CORS spec forbids it.
    # List every header the frontend actually sends.
    allow_headers     = [
        "Content-Type",
        "Accept",
        "Origin",
        "Authorization",
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL EXCEPTION HANDLER
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception on {request.url.path}: {exc}")
    log.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={"error_code": "INTERNAL_ERROR", "message": "Something went wrong. Please try again."},
    )


# ─────────────────────────────────────────────────────────────────────────────
# SESSION COOKIE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_session_id_from_cookie(request: Request) -> str | None:
    """Read session_id from the cookie. Returns None if missing."""
    return request.cookies.get(settings.session_cookie_name)


def _set_session_cookie(response: Response, session_id: str) -> None:
    """Attach the session cookie to any response."""
    response.set_cookie(
        key      = settings.session_cookie_name,
        value    = session_id,
        max_age  = settings.session_cookie_max_age,
        httponly = True,                         # JS cannot read — XSS protection
        secure   = settings.session_cookie_secure,
        samesite = settings.session_cookie_samesite,
        path     = "/",
    )


def _require_session_id(request: Request) -> str:
    """
    FastAPI dependency — extracts session_id from cookie.
    Raises 401 if cookie is missing so routes stay clean.
    """
    sid = _get_session_id_from_cookie(request)
    if not sid:
        raise HTTPException(status_code=401, detail={
            "error_code": "NO_SESSION",
            "message": "No active session found. Please start a new session.",
        })
    return sid


def _require_session_row(request: Request) -> dict:
    """
    FastAPI dependency — resolves session_id cookie to a raw DB row.
    Raises 401 if cookie missing, 404 if session not found.
    """
    sid = _require_session_id(request)
    row = session_svc.get_session_raw(sid)
    if not row:
        raise HTTPException(status_code=404, detail={
            "error_code": "SESSION_NOT_FOUND",
            "message": "Session not found or expired. Please start a new session.",
        })
    return row


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Aina backend is live!"}

@app.get("/health")
async def health():
    """Liveness probe."""
    return {"status": "ok", "service": "aina-main-backend"}


# ── POST /session ─────────────────────────────────────────────────────────────

@app.post(
    "/session",
    response_model = SessionResponse,
    status_code    = 201,
    summary        = "Create a new anonymous session",
    description    = (
        "Called once at Step 2 when the user has confirmed gender + height + weight. "
        "Sets an HttpOnly session cookie used by all subsequent requests. "
        "If a valid session cookie already exists, patches it with the new values instead of creating a duplicate."
    ),
)
async def create_session(
    body:     SessionCreateRequest,
    response: Response,
    request:  Request,
):
    # If there's already a valid session cookie, reuse it
    existing_sid = _get_session_id_from_cookie(request)
    if existing_sid:
        existing = session_svc.get_session_raw(existing_sid)
        if existing:
            log.info(f"Reusing existing session: {existing_sid}")
            # Update gender/height/weight in case user went back and changed them
            session_svc._patch(existing_sid, {
                "gender":    body.gender.value,
                "height_cm": body.height_cm,
                "weight_kg": body.weight_kg,
            })
            updated = session_svc.get_session(existing_sid)
            _set_session_cookie(response, existing_sid)
            # 200 OK — this is an update, not a creation
            response.status_code = 200
            return updated

    session = session_svc.create_session(body)
    _set_session_cookie(response, session.session_id)
    log.info(f"New session created: {session.session_id}")
    return session


# ── GET /session ──────────────────────────────────────────────────────────────

@app.get(
    "/session",
    response_model = SessionResponse,
    summary        = "Get current session state",
    description    = "Read the current session from the cookie. Useful for resuming a flow.",
)
async def get_session(session_row: dict = Depends(_require_session_row)):
    return session_svc._row_to_response(session_row)


# ── POST /session/face-photo ──────────────────────────────────────────────────

@app.post(
    "/session/face-photo",
    response_model = UploadFaceResponse,
    summary        = "Upload face photo",
    description    = (
        "Step 3 — upload the shopper's face photo. "
        "Stored in Supabase Storage. Returns a signed URL for preview."
    ),
)
async def upload_face_photo(
    file:        UploadFile = File(...),
    session_row: dict       = Depends(_require_session_row),
):
    session_id = session_row["session_id"]
    log.info(f"[{session_id}] Face photo upload — {file.filename} ({file.content_type})")

    storage_path, signed_url = await storage_svc.upload_face_photo(session_id, file)
    session_svc.set_face_photo(session_id, storage_path)

    return UploadFaceResponse(
        session_id     = session_id,
        face_photo_url = signed_url,
        status         = SessionStatus.images_uploaded,
    )


# ── POST /session/extract ─────────────────────────────────────────────────────

@app.post(
    "/session/extract",
    response_model = ExtractResponse,
    summary        = "Upload body photo and extract measurements",
    description    = (
        "Step 4 — single endpoint that accepts the body photo, stores it in Supabase Storage, "
        "then immediately forwards it to the GPU worker for AI measurement extraction. "
        "No separate upload step required. "
        "Idempotent on re-submit: if the session already has completed measurements, "
        "the new photo replaces the old one and re-runs the worker."
    ),
)
async def extract_measurements(
    file:        UploadFile = File(..., description="Full-body photo (JPG, PNG, WebP, max 10MB)"),
    session_row: dict       = Depends(_require_session_row),
):
    session_id = session_row["session_id"]
    log.info(f"[{session_id}] Extract request — {file.filename} ({file.content_type})")

    # ── Guard: don't allow concurrent processing ──────────────────────────────
    if session_row["status"] == SessionStatus.processing.value:
        raise HTTPException(status_code=409, detail={
            "error_code": "ALREADY_PROCESSING",
            "message": "Your measurements are already being processed. Please wait.",
        })

    # ── Upload body photo + capture bytes in one operation ───────────────────
    # upload_body_photo reads the stream once and returns (path, signed_url, bytes).
    # We use those same bytes for the worker call — no second network trip needed.
    storage_path, _, image_bytes = await storage_svc.upload_body_photo(session_id, file)
    session_svc.set_body_photo(session_id, storage_path)

    # ── Mark as processing ────────────────────────────────────────────────────
    session_svc.set_processing(session_id)

    # ── Forward bytes directly to GPU worker ─────────────────────────────────
    filename = storage_path.split("/")[-1]

    # ── Call GPU worker ───────────────────────────────────────────────────────
    try:
        measurements = await worker_svc.call_extract_3d(
            session_id  = session_id,
            image_bytes = image_bytes,
            filename    = filename,
            height_cm   = session_row["height_cm"],
            weight_kg   = session_row["weight_kg"],
            gender      = session_row["gender"],
        )
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, dict) else {"error_code": "FAILED", "message": str(e.detail)}
        session_svc.set_failed(
            session_id,
            error_code = detail.get("error_code", "FAILED"),
            message    = detail.get("message", "Processing failed."),
        )
        raise

    # ── Persist measurements and return ──────────────────────────────────────
    session_svc.set_measurements(session_id, measurements.model_dump())

    return ExtractResponse(
        session_id   = session_id,
        status       = SessionStatus.completed,
        measurements = measurements,
    )


# ── PATCH /session/measurements ───────────────────────────────────────────────

@app.patch(
    "/session/measurements",
    response_model = MeasurementUpdateResponse,
    summary        = "Persist user-edited measurements",
    description    = (
        "Called from Step 5 when the user confirms their measurements (edited or not). "
        "Overwrites the four editable fields — chest, waist, hip, shoulder — in the "
        "stored measurements object. bmi is always preserved from the original AI "
        "extraction and is never overwritten by this endpoint. "
        "Must be called before POST /session/avatar so that avatar generation and "
        "all downstream size recommendations use the user-confirmed values, not the "
        "raw AI output."
    ),
)
async def update_measurements(
    body:        MeasurementUpdateRequest,
    session_row: dict = Depends(_require_session_row),
):
    session_id = session_row["session_id"]

    # Guard: measurements must already exist (extract must have run first)
    if not session_row.get("measurements"):
        raise HTTPException(status_code=422, detail={
            "error_code": "NO_MEASUREMENTS",
            "message":    "No measurements found. Please complete the body photo analysis first.",
        })

    # Snapshot the current AI-output measurements BEFORE overwriting them.
    # This is the ground-truth we're scoring against — once update_measurements()
    # runs, the original AI values are gone from the sessions table.
    ai_measurements = dict(session_row["measurements"])

    user_patch = {
        "chest_cm":          body.chest_cm,
        "waist_cm":          body.waist_cm,
        "hip_cm":            body.hip_cm,
        "shoulder_width_cm": body.shoulder_width_cm,
    }

    try:
        merged = session_svc.update_measurements(session_id, patch=user_patch)
    except RuntimeError as e:
        log.error(str(e))
        raise HTTPException(status_code=500, detail={
            "error_code": "MEASUREMENTS_UPDATE_FAILED",
            "message":    "Failed to save your measurements. Please try again.",
        })

    # Record the correction event for calibration. Non-fatal — runs after the
    # session update so a DB write failure here never blocks the user response.
    correction_svc.record_correction(
        session_id      = session_id,
        session_row     = session_row,
        ai_measurements = ai_measurements,
        user_patch      = user_patch,
    )

    log.info(f"[{session_id}] User-confirmed measurements persisted")
    return MeasurementUpdateResponse(
        session_id   = session_id,
        measurements = Measurements(**merged),
    )


# ── POST /session/avatar ──────────────────────────────────────────────────────

@app.post(
    "/session/avatar",
    response_model = AvatarResponse,
    summary        = "Generate digital avatar from body photo",
    description    = (
        "Generates a photorealistic full-body digital avatar of the user "
        "using their stored body photo. The avatar preserves face, skin tone, "
        "and body proportions dressed in fitted black gym wear in A-pose. "
        "After successful generation the raw body and face photos are permanently "
        "deleted from storage — only the avatar is retained. "
        "The avatar is reused for all subsequent try-on requests. "
        "Idempotent: if avatar already exists returns the stored signed URL immediately."
    ),
)
async def generate_avatar(
    session_row: dict = Depends(_require_session_row),
):
    session_id = session_row["session_id"]

    # ── Idempotency — return existing avatar immediately ──────────────────────
    if (
        session_row.get("avatar_status") == AvatarStatus.ready.value
        and session_row.get("avatar_path")
    ):
        log.info(f"[{session_id}] Returning cached avatar")
        signed_url = storage_svc.get_signed_url(
            settings.storage_bucket_avatar,
            session_row["avatar_path"],
        )
        return AvatarResponse(
            session_id    = session_id,
            avatar_status = AvatarStatus.ready,
            avatar_url    = signed_url,
            provider      = session_row.get("avatar_provider", "gemini"),
        )

    # ── Guard: measurements must exist before generating avatar ───────────────
    if not session_row.get("measurements"):
        raise HTTPException(status_code=422, detail={
            "error_code": "MEASUREMENTS_REQUIRED",
            "message":    "Please complete measurement extraction before generating your avatar.",
        })

    # ── Guard: stale-generating recovery ─────────────────────────────────────
    # This MUST come before the body-photo guard.
    #
    # Scenario that causes the bug without this block:
    #   1. Client calls POST /session/avatar — backend sets avatar_status=generating
    #   2. Client times out or aborts after 90s
    #   3. Backend Vertex call finishes (success or failure) and updates the DB
    #      OR the backend process itself crashed, leaving status permanently stuck
    #   4. Client retries — hits the body-photo guard (which comes next) BEFORE
    #      the generating guard, so it gets 422 "Body photo missing" when the
    #      real problem is a stale generating state, not a missing photo
    #
    # Fix: check for generating FIRST. If it has been generating for >120s
    # (well past the Vertex 90s ceiling), treat it as a stale lock and reset
    # to failed so the retry can proceed cleanly.
    #
    # If it's been generating for <120s the previous request is still legitimately
    # in-flight — return 409 so the client knows to wait and poll.
    if session_row.get("avatar_status") == AvatarStatus.generating.value:
        STALE_GENERATING_SECONDS = 120
        updated_at_raw = session_row.get("updated_at")
        is_stale = False
        if updated_at_raw:
            from datetime import datetime, timezone as _tz
            try:
                if isinstance(updated_at_raw, str):
                    updated_at = datetime.fromisoformat(updated_at_raw.replace("Z", "+00:00"))
                else:
                    updated_at = updated_at_raw
                age_seconds = (datetime.now(_tz.utc) - updated_at).total_seconds()
                is_stale = age_seconds > STALE_GENERATING_SECONDS
            except Exception:
                is_stale = False  # be conservative — don't reset if we can't parse

        if is_stale:
            log.warning(
                f"[{session_id}] Stale generating lock detected "
                f"(updated_at={updated_at_raw}) — resetting to failed so retry can proceed"
            )
            session_svc.set_avatar_failed(
                session_id,
                error_code = "STALE_GENERATING_RESET",
                message    = "Previous generation attempt timed out.",
            )
            # Fall through — let the retry proceed past this guard
        else:
            raise HTTPException(status_code=409, detail={
                "error_code": "AVATAR_ALREADY_GENERATING",
                "message":    "Your avatar is already being generated. Please wait.",
            })

    # ── Guard: body photo must still exist ────────────────────────────────────
    # This guard comes AFTER the generating check deliberately. A stale-generating
    # session may have had its body_photo_path nulled if the server completed
    # avatar generation but the client never received the response. In that case
    # avatar_status will be 'ready' (caught by idempotency above) or 'failed' with
    # the body photo still intact. The only time body_photo_path is None with a
    # non-ready status is a genuine re-upload requirement.
    body_photo_path = session_row.get("body_photo_path")
    if not body_photo_path:
        raise HTTPException(status_code=422, detail={
            "error_code": "BODY_PHOTO_MISSING",
            "message":    "Body photo not found. Please re-upload your photo.",
        })

    # ── Download body photo from storage ─────────────────────────────────────
    body_image_bytes = await storage_svc.download_body_photo(session_id, body_photo_path)

    # Infer mime type from stored path extension
    body_mime = "image/jpeg"
    if body_photo_path.endswith(".png"):
        body_mime = "image/png"
    elif body_photo_path.endswith(".webp"):
        body_mime = "image/webp"

    # ── Mark as generating ────────────────────────────────────────────────────
    session_svc.set_avatar_generating(session_id)

    # ── Generate avatar via configured provider ───────────────────────────────
    try:
        avatar_bytes, provider = await avatar_svc.generate_avatar(
            session_id        = session_id,
            body_image_bytes  = body_image_bytes,
            body_image_mime   = body_mime,
            gender            = session_row["gender"],
        )
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, dict) else {
            "error_code": "AVATAR_FAILED", "message": str(e.detail)
        }
        session_svc.set_avatar_failed(
            session_id,
            error_code = detail.get("error_code", "AVATAR_FAILED"),
            message    = detail.get("message", "Avatar generation failed."),
        )
        raise

    # ── Upload avatar to storage ──────────────────────────────────────────────
    avatar_path, signed_url = await storage_svc.upload_avatar_bytes(
        session_id  = session_id,
        image_bytes = avatar_bytes,
        content_type = "image/jpeg",
    )

    # ── Persist avatar, clear raw photos from DB ──────────────────────────────
    session_svc.set_avatar_ready(session_id, avatar_path, provider)
    session_svc.clear_raw_photos(session_id)

    # ── Delete raw photos from storage (non-blocking, non-fatal) ─────────────
    storage_svc.delete_session_photos(
        session_id      = session_id,
        face_photo_path = session_row.get("face_photo_path"),
        body_photo_path = body_photo_path,
    )

    log.info(f"[{session_id}] Avatar complete via {provider} — raw photos deleted")

    return AvatarResponse(
        session_id    = session_id,
        avatar_status = AvatarStatus.ready,
        avatar_url    = signed_url,
        provider      = provider,
    )


# ── DELETE /session ───────────────────────────────────────────────────────────

@app.post(
    "/session/reset",
    summary     = "Reset (clear) the current session cookie",
    description = "Clears the session cookie. The DB row is kept for analytics.",
)
async def reset_session(response: Response):
    response.delete_cookie(
        key      = settings.session_cookie_name,
        path     = "/",
        httponly = True,
        secure   = settings.session_cookie_secure,
        samesite = settings.session_cookie_samesite,
    )
    return {"status": "ok", "message": "Session cleared."}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = not settings.is_production,
        workers = 1,          # single worker — GPU calls are inherently serial
    )