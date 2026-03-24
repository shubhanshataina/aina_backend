# services/storage_service.py
# Handles uploading images to Supabase Storage and generating signed URLs.
# All paths follow: {session_id}/{image_type}_{timestamp}.{ext}

import logging
import mimetypes
from datetime import datetime, timezone

from fastapi import UploadFile, HTTPException
from db.supabase_client import get_supabase
from config import get_settings

log = logging.getLogger("app")

# Signed URL TTL — 1 hour is enough for the frontend to display a preview.
# The path is what we store permanently; signed URLs are ephemeral.
SIGNED_URL_TTL_SECONDS = 3600

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

async def upload_avatar_bytes(
    session_id:   str,
    image_bytes:  bytes,
    content_type: str = "image/jpeg",
) -> tuple[str, str]:
    """
    Upload generated avatar bytes directly to Supabase Storage.
    Unlike face/body uploads this takes raw bytes, not an UploadFile,
    because the avatar comes from an API response, not a browser upload.

    Returns:
        (storage_path, signed_url)
    """
    settings  = get_settings()
    bucket    = settings.storage_bucket_avatar
    ext       = _extension_from_mime(content_type)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path      = f"{session_id}/avatar_{timestamp}{ext}"

    try:
        get_supabase().storage.from_(bucket).upload(
            path         = path,
            file         = image_bytes,
            file_options = {"content-type": content_type, "upsert": "true"},
        )
        log.info(f"[{session_id}] Uploaded avatar → {bucket}/{path} ({len(image_bytes):,} bytes)")
    except Exception as e:
        log.error(f"[{session_id}] Avatar storage upload failed: {e}")
        raise HTTPException(status_code=500, detail={
            "error_code": "AVATAR_STORAGE_FAILED",
            "message":    "Failed to save your avatar. Please try again.",
        })

    try:
        signed_url = get_signed_url(bucket, path)
    except Exception as e:
        log.warning(f"[{session_id}] Avatar signed URL failed (non-fatal): {e}")
        signed_url = ""

    return path, signed_url


async def upload_face_photo(session_id: str, file: UploadFile) -> tuple[str, str]:
    """
    Validate, upload face photo to Supabase Storage, return signed preview URL.

    Returns:
        (storage_path, signed_url)
        storage_path — persisted in DB permanently
        signed_url   — short-lived URL safe to return to the client for preview
    """
    path, signed_url, _ = await _upload(
        session_id = session_id,
        file       = file,
        image_type = "face",
        bucket     = get_settings().storage_bucket_face,
    )
    return path, signed_url


async def upload_body_photo(session_id: str, file: UploadFile) -> tuple[str, str, bytes]:
    """
    Validate, upload body photo to Supabase Storage, and return the raw bytes.

    Returns:
        (storage_path, signed_url, image_bytes)
        storage_path — persisted in DB permanently
        signed_url   — short-lived URL (not used by extract, kept for consistency)
        image_bytes  — the exact bytes that were uploaded, ready to forward to
                       the GPU worker without a second network round-trip.

    The bytes are read once, used for both the Supabase upload and the worker
    call — no upload-then-download cycle.
    """
    return await _upload(
        session_id = session_id,
        file       = file,
        image_type = "body",
        bucket     = get_settings().storage_bucket_body,
    )


def get_signed_url(bucket: str, path: str, ttl: int = SIGNED_URL_TTL_SECONDS) -> str:
    """Generate a short-lived signed URL for a stored object."""
    result = get_supabase().storage.from_(bucket).create_signed_url(path, ttl)
    if "signedURL" not in result:
        raise RuntimeError(f"Failed to generate signed URL for {path}: {result}")
    return result["signedURL"]


async def download_body_photo(session_id: str, body_photo_path: str) -> bytes:
    """
    Download body photo bytes from Supabase Storage.
    Called by /session/avatar to fetch the stored body photo before
    forwarding to the avatar generation service.
    """
    settings = get_settings()
    try:
        data = (
            get_supabase()
            .storage
            .from_(settings.storage_bucket_body)
            .download(body_photo_path)
        )
        return data
    except Exception as e:
        log.error(f"[{session_id}] Failed to download body photo: {e}")
        raise HTTPException(status_code=500, detail={
            "error_code": "STORAGE_DOWNLOAD_FAILED",
            "message":    "Could not retrieve your photo. Please try uploading again.",
        })


def delete_session_photos(
    session_id:      str,
    face_photo_path: str | None,
    body_photo_path: str | None,
) -> None:
    """
    Delete raw face and body photos from Supabase Storage after avatar
    generation is complete. Non-fatal per file — logs failures but does
    not raise so the avatar response is never blocked by a cleanup error.

    Called exclusively from the avatar route after avatar_url is safely stored.
    """
    settings = get_settings()

    def _delete(bucket: str, path: str | None) -> None:
        if not path:
            return
        try:
            get_supabase().storage.from_(bucket).remove([path])
            log.info(f"[{session_id}] Deleted {bucket}/{path}")
        except Exception as e:
            # Non-fatal: log and continue — analytics row still has measurements
            log.warning(f"[{session_id}] Failed to delete {bucket}/{path}: {e}")

    _delete(settings.storage_bucket_face, face_photo_path)
    _delete(settings.storage_bucket_body, body_photo_path)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _upload(
    session_id: str,
    file:       UploadFile,
    image_type: str,
    bucket:     str,
) -> tuple[str, str, bytes]:
    """
    Core upload logic shared by face and body uploads.

    Reads the file stream exactly once into memory, validates mime type and
    size, builds a timestamped storage path, uploads to Supabase Storage,
    and returns (path, signed_url, raw_bytes).

    Returning raw_bytes means callers never need to re-download what was just
    uploaded — the bytes are available immediately for any downstream use
    (e.g. forwarding to the GPU worker).
    """

    # ── Validate mime type ────────────────────────────────────────────────────
    content_type = file.content_type or ""
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=422, detail={
            "error_code": "INVALID_FILE_TYPE",
            "message": "Please upload a JPG, PNG, or WebP image.",
        })

    # ── Single read — used for validation, upload, and worker forwarding ──────
    contents = await file.read()

    if len(contents) == 0:
        raise HTTPException(status_code=422, detail={
            "error_code": "EMPTY_FILE",
            "message": "The uploaded file is empty.",
        })
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=422, detail={
            "error_code": "FILE_TOO_LARGE",
            "message": "Image must be under 10 MB.",
        })

    # ── Build storage path ────────────────────────────────────────────────────
    ext       = _extension_from_mime(content_type)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path      = f"{session_id}/{image_type}_{timestamp}{ext}"

    # ── Upload ────────────────────────────────────────────────────────────────
    try:
        get_supabase().storage.from_(bucket).upload(
            path         = path,
            file         = contents,
            file_options = {"content-type": content_type, "upsert": "true"},
        )
        log.info(f"[{session_id}] Uploaded {image_type} photo → {bucket}/{path} ({len(contents):,} bytes)")
    except Exception as e:
        log.error(f"[{session_id}] Storage upload failed: {e}")
        raise HTTPException(status_code=500, detail={
            "error_code": "STORAGE_UPLOAD_FAILED",
            "message": "Failed to store your image. Please try again.",
        })

    # ── Signed URL (non-fatal if it fails) ────────────────────────────────────
    try:
        signed_url = get_signed_url(bucket, path)
    except Exception as e:
        log.warning(f"[{session_id}] Signed URL generation failed (non-fatal): {e}")
        signed_url = ""

    # Return path + signed_url + the bytes we already have in memory
    return path, signed_url, contents


def _extension_from_mime(mime: str) -> str:
    ext = mimetypes.guess_extension(mime)
    # mimetypes can return .jpe for image/jpeg — normalise
    return {".jpe": ".jpg", ".jpeg": ".jpg"}.get(ext, ext) if ext else ".jpg"