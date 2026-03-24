# services/session_service.py
# All session read/write operations against Supabase Postgres.
# This is the only file that talks directly to the `sessions` table.

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from db.supabase_client import get_supabase
from models.schemas import (
    Gender, SessionStatus, AvatarStatus, Measurements,
    SessionCreateRequest, SessionResponse,
)

log = logging.getLogger("app")

TABLE = "sessions"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _row_to_response(row: dict) -> SessionResponse:
    """Convert a raw Supabase row dict into a safe public SessionResponse."""
    measurements = None
    if row.get("measurements"):
        measurements = Measurements(**row["measurements"])

    # avatar_status defaults to pending if column not yet populated
    raw_avatar_status = row.get("avatar_status") or AvatarStatus.pending.value

    return SessionResponse(
        session_id     = row["session_id"],
        status         = SessionStatus(row["status"]),
        gender         = Gender(row["gender"]),
        height_cm      = row["height_cm"],
        weight_kg      = row["weight_kg"],
        has_face_photo = bool(row.get("face_photo_path")),
        has_body_photo = bool(row.get("body_photo_path")),
        has_avatar     = bool(row.get("avatar_path")),
        avatar_status  = AvatarStatus(raw_avatar_status),
        measurements   = measurements,
        error_message  = row.get("error_message"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def create_session(data: SessionCreateRequest) -> SessionResponse:
    """
    Insert a new session row and return it.
    session_id is a UUID generated here — never from the client.
    """
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    row = {
        "session_id": session_id,
        "status":     SessionStatus.initiated.value,
        "gender":     data.gender.value,
        "height_cm":  data.height_cm,
        "weight_kg":  data.weight_kg,
        "created_at": now,
        "updated_at": now,
    }

    result = get_supabase().table(TABLE).insert(row).execute()

    if not result.data:
        raise RuntimeError(f"Failed to create session: {result}")

    log.info(f"[{session_id}] Session created gender={data.gender} h={data.height_cm} w={data.weight_kg}")
    return _row_to_response(result.data[0])


def get_session(session_id: str) -> Optional[SessionResponse]:
    """Fetch a session by ID. Returns None if not found."""
    result = (
        get_supabase()
        .table(TABLE)
        .select("*")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    if not result.data:
        return None
    return _row_to_response(result.data)


def get_session_raw(session_id: str) -> Optional[dict]:
    """
    Like get_session but returns the raw row dict.
    Used internally by services that need full fields (e.g. storage paths).
    """
    result = (
        get_supabase()
        .table(TABLE)
        .select("*")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    return result.data or None


def set_face_photo(session_id: str, storage_path: str) -> None:
    """Record the Supabase Storage path for the face photo."""
    _patch(session_id, {
        "face_photo_path": storage_path,
        "status": SessionStatus.images_uploaded.value,
    })
    log.info(f"[{session_id}] Face photo recorded: {storage_path}")


def set_body_photo(session_id: str, storage_path: str) -> None:
    """Record the Supabase Storage path for the body photo."""
    _patch(session_id, {
        "body_photo_path": storage_path,
        "status": SessionStatus.images_uploaded.value,
    })
    log.info(f"[{session_id}] Body photo recorded: {storage_path}")


def set_processing(session_id: str) -> None:
    """Mark session as in-flight while worker processes it."""
    _patch(session_id, {"status": SessionStatus.processing.value})
    log.info(f"[{session_id}] Status → processing")


def set_measurements(session_id: str, measurements: dict) -> None:
    """Store worker results and mark session completed."""
    _patch(session_id, {
        "measurements": measurements,
        "status":       SessionStatus.completed.value,
        "error_message": None,
    })
    log.info(f"[{session_id}] Measurements stored, status → completed")


def update_measurements(session_id: str, patch: dict) -> dict:
    """
    Overwrite only the four user-editable measurement fields.
    bmi is preserved from the existing row — it is derived from height/weight
    and must not be overwritten by client-supplied values.

    Args:
        patch — dict with keys: chest_cm, waist_cm, hip_cm, shoulder_width_cm

    Returns:
        The full updated measurements dict (including preserved bmi).

    Raises:
        RuntimeError if the session has no existing measurements to patch.
    """
    row = get_session_raw(session_id)
    if not row or not row.get("measurements"):
        raise RuntimeError(f"[{session_id}] Cannot update measurements: no existing measurements found.")

    # Merge: keep bmi from the AI extraction, overwrite only the four editable fields.
    merged = {**row["measurements"], **patch}

    _patch(session_id, {"measurements": merged})
    log.info(
        f"[{session_id}] Measurements updated by user — "
        f"chest={patch.get('chest_cm')} waist={patch.get('waist_cm')} "
        f"hip={patch.get('hip_cm')} shoulder={patch.get('shoulder_width_cm')}"
    )
    return merged


def set_failed(session_id: str, error_code: str, message: str) -> None:
    """Mark session as failed with a sanitised error message."""
    _patch(session_id, {
        "status":        SessionStatus.failed.value,
        "error_message": f"{error_code}: {message}",
    })
    log.warning(f"[{session_id}] Status → failed ({error_code})")


def set_avatar_generating(session_id: str) -> None:
    """Mark avatar generation as in-flight."""
    _patch(session_id, {"avatar_status": AvatarStatus.generating.value})
    log.info(f"[{session_id}] Avatar status → generating")


def set_avatar_ready(session_id: str, avatar_path: str, provider: str) -> None:
    """
    Store the generated avatar path and mark it ready.
    provider — which service generated it: 'gemini' | 'fal' | 'replicate'
    Stored for analytics and future migration tracking.
    """
    _patch(session_id, {
        "avatar_path":     avatar_path,
        "avatar_status":   AvatarStatus.ready.value,
        "avatar_provider": provider,
    })
    log.info(f"[{session_id}] Avatar ready via {provider}: {avatar_path}")


def set_avatar_failed(session_id: str, error_code: str, message: str) -> None:
    """Mark avatar generation as failed — does not affect session status."""
    _patch(session_id, {
        "avatar_status":   AvatarStatus.failed.value,
        "error_message":   f"AVATAR_{error_code}: {message}",
    })
    log.warning(f"[{session_id}] Avatar failed ({error_code})")


def clear_raw_photos(session_id: str) -> None:
    """
    Null out storage paths for face and body photos after avatar is generated.
    The files are deleted from Supabase Storage separately in storage_service.
    This ensures the DB reflects what's actually in storage.
    """
    _patch(session_id, {
        "face_photo_path": None,
        "body_photo_path": None,
    })
    log.info(f"[{session_id}] Raw photo paths cleared from session")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _patch(session_id: str, fields: dict) -> None:
    """Apply a partial update to a session row."""
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    get_supabase().table(TABLE).update(fields).eq("session_id", session_id).execute()