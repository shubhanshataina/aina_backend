# services/worker_service.py
# Internal-only HTTP client that forwards requests to the GPU worker (hmr_worker.py).
# Never called directly by the frontend — always mediated through main.py routes.
#
# Uses aiohttp instead of httpx to avoid the httpx version conflict between
# supabase (requires httpx<0.29) and google-genai (requires httpx>=0.28.1).
# aiohttp has no overlap with either constraint.

import logging
import aiohttp
from fastapi import HTTPException

from config import get_settings
from models.schemas import Measurements

log = logging.getLogger("app")

# Timeout config — HMR2 inference takes 8–15s, so we give it generous headroom.
# aiohttp.ClientTimeout uses total/connect/sock_read/sock_connect fields.
WORKER_TIMEOUT = aiohttp.ClientTimeout(
    total        = 95.0,  # overall request ceiling
    connect      =  5.0,  # TCP connection to worker
    sock_read    = 60.0,  # wait up to 60s for inference result
    sock_connect =  5.0,
)


async def call_extract_3d(
    session_id:  str,
    image_bytes: bytes,
    filename:    str,
    height_cm:   float,
    weight_kg:   float,
    gender:      str,
) -> Measurements:
    """
    POST the body image + anthropometrics to the GPU worker's /extract_3d endpoint.

    Args:
        session_id  — used only for logging/tracing
        image_bytes — raw bytes of the body photo (fetched from Supabase Storage)
        filename    — original filename, forwarded to worker
        height_cm   — from session
        weight_kg   — from session
        gender      — from session ("male" | "female")

    Returns:
        Measurements pydantic model with all extracted measurements.

    Raises:
        HTTPException — with sanitised error_code/message on any failure.
                        The raw worker error is logged but never forwarded to the client.
    """
    settings   = get_settings()
    worker_url = settings.worker_url.rstrip("/") + "/extract_3d"

    # Determine content type from filename extension
    ct = "image/jpeg"
    if filename.lower().endswith(".png"):
        ct = "image/png"
    elif filename.lower().endswith(".webp"):
        ct = "image/webp"

    log.info(f"[{session_id}] Calling worker: {worker_url} h={height_cm} w={weight_kg} g={gender}")

    # aiohttp multipart: FormData handles both text fields and file uploads.
    # We build it explicitly so the field names match what hmr_worker expects.
    data = aiohttp.FormData()
    data.add_field("height_cm", str(height_cm))
    data.add_field("weight_kg", str(weight_kg))
    data.add_field("gender",    gender)
    data.add_field(
        "file",
        image_bytes,
        filename     = filename,
        content_type = ct,
    )

    try:
        async with aiohttp.ClientSession(timeout=WORKER_TIMEOUT) as client:
            async with client.post(worker_url, data=data) as response:
                # Read the body inside the context manager before the
                # connection is released back to the pool.
                response_text   = await response.text()
                response_status = response.status
    except aiohttp.ClientConnectorError:
        log.error(f"[{session_id}] Worker connection refused at {worker_url}")
        raise HTTPException(status_code=503, detail={
            "error_code": "WORKER_UNAVAILABLE",
            "message": "Measurement service is temporarily unavailable. Please try again in a moment.",
        })
    except aiohttp.ServerTimeoutError:
        log.error(f"[{session_id}] Worker timed out after {WORKER_TIMEOUT.sock_read}s read timeout")
        raise HTTPException(status_code=504, detail={
            "error_code": "WORKER_TIMEOUT",
            "message": "Analysis is taking longer than expected. Please try again.",
        })
    except Exception as e:
        log.error(f"[{session_id}] Unexpected worker error: {e}")
        raise HTTPException(status_code=500, detail={
            "error_code": "WORKER_ERROR",
            "message": "Processing failed. Please try again.",
        })

    # ── Parse worker response ─────────────────────────────────────────────────
    import json as _json
    try:
        payload = _json.loads(response_text)
    except Exception:
        log.error(f"[{session_id}] Worker returned non-JSON response: {response_text[:200]}")
        raise HTTPException(status_code=500, detail={
            "error_code": "WORKER_BAD_RESPONSE",
            "message": "Processing failed. Please try again.",
        })

    if response_status != 200:
        # Worker returned a structured error — forward the error_code and message
        # but log the full detail internally
        detail = payload.get("detail", {})
        error_code = detail.get("error_code", f"WORKER_HTTP_{response_status}") if isinstance(detail, dict) else f"WORKER_HTTP_{response_status}"
        message    = detail.get("message",    "Processing failed. Please try again.") if isinstance(detail, dict) else str(detail)

        log.warning(f"[{session_id}] Worker error {response_status}: {error_code} — {message}")

        # Propagate validation errors (422) directly — these are user-facing
        # (e.g. bad photo, body not visible). Other errors are sanitised.
        if response_status == 422:
            raise HTTPException(status_code=422, detail={
                "error_code": error_code,
                "message":    message,
            })
        raise HTTPException(status_code=500, detail={
            "error_code": "PROCESSING_FAILED",
            "message": "Could not extract measurements. Please try again with a clearer photo.",
        })

    # ── Extract measurements from success response ─────────────────────────
    raw_measurements = payload.get("measurements")
    if not raw_measurements:
        log.error(f"[{session_id}] Worker success response missing measurements: {payload}")
        raise HTTPException(status_code=500, detail={
            "error_code": "WORKER_MISSING_DATA",
            "message": "Processing failed. Please try again.",
        })

    try:
        measurements = Measurements(**raw_measurements)
    except Exception as e:
        log.error(f"[{session_id}] Measurements parse error: {e} — raw: {raw_measurements}")
        raise HTTPException(status_code=500, detail={
            "error_code": "MEASUREMENTS_PARSE_FAILED",
            "message": "Processing failed. Please try again.",
        })

    log.info(
        f"[{session_id}] Worker success — "
        f"chest={measurements.chest_cm} waist={measurements.waist_cm} "
        f"hip={measurements.hip_cm} shoulder={measurements.shoulder_width_cm} bmi={measurements.bmi}"
    )
    return measurements