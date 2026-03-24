# services/worker_service.py
# Internal-only HTTP client that forwards requests to the GPU worker (hmr_worker.py).
# Never called directly by the frontend — always mediated through main.py routes.

import logging
import httpx
from fastapi import HTTPException

from config import get_settings
from models.schemas import Measurements

log = logging.getLogger("app")

# Timeout config — HMR2 inference takes 8–15s, so we give it generous headroom.
WORKER_TIMEOUT = httpx.Timeout(
    connect =  5.0,   # connection to worker
    read    = 60.0,   # wait up to 60s for inference result
    write   = 30.0,   # uploading the image to worker
    pool    =  5.0,
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

    form_data = {
        "height_cm": str(height_cm),
        "weight_kg": str(weight_kg),
        "gender":    gender,
    }
    files = {
        "file": (filename, image_bytes, ct),
    }

    log.info(f"[{session_id}] Calling worker: {worker_url} h={height_cm} w={weight_kg} g={gender}")

    try:
        async with httpx.AsyncClient(timeout=WORKER_TIMEOUT) as client:
            response = await client.post(worker_url, data=form_data, files=files)
    except httpx.ConnectError:
        log.error(f"[{session_id}] Worker connection refused at {worker_url}")
        raise HTTPException(status_code=503, detail={
            "error_code": "WORKER_UNAVAILABLE",
            "message": "Measurement service is temporarily unavailable. Please try again in a moment.",
        })
    except httpx.TimeoutException:
        log.error(f"[{session_id}] Worker timed out after {WORKER_TIMEOUT.read}s")
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
    try:
        payload = response.json()
    except Exception:
        log.error(f"[{session_id}] Worker returned non-JSON response: {response.text[:200]}")
        raise HTTPException(status_code=500, detail={
            "error_code": "WORKER_BAD_RESPONSE",
            "message": "Processing failed. Please try again.",
        })

    if response.status_code != 200:
        # Worker returned a structured error — forward the error_code and message
        # but log the full detail internally
        detail = payload.get("detail", {})
        error_code = detail.get("error_code", f"WORKER_HTTP_{response.status_code}") if isinstance(detail, dict) else f"WORKER_HTTP_{response.status_code}"
        message    = detail.get("message",    "Processing failed. Please try again.") if isinstance(detail, dict) else str(detail)

        log.warning(f"[{session_id}] Worker error {response.status_code}: {error_code} — {message}")

        # Propagate validation errors (422) directly — these are user-facing
        # (e.g. bad photo, body not visible). Other errors are sanitised.
        if response.status_code == 422:
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