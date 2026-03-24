# models/schemas.py — all Pydantic models for request/response validation

from __future__ import annotations
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Gender(str, Enum):
    male   = "male"
    female = "female"


class SessionStatus(str, Enum):
    initiated        = "initiated"       # session created, no images yet
    images_uploaded  = "images_uploaded" # images stored, not yet processed
    processing       = "processing"      # worker call in-flight
    completed        = "completed"       # measurements extracted successfully
    failed           = "failed"          # worker returned an error


class AvatarStatus(str, Enum):
    pending     = "pending"      # not yet requested
    generating  = "generating"   # Gemini call in-flight
    ready       = "ready"        # avatar image stored
    failed      = "failed"       # generation failed


# ─────────────────────────────────────────────────────────────────────────────
# Measurements
# ─────────────────────────────────────────────────────────────────────────────

class Measurements(BaseModel):
    chest_cm:          float
    waist_cm:          float
    hip_cm:            float
    shoulder_width_cm: float
    bmi:               float


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────

class SessionCreateRequest(BaseModel):
    gender:    Gender
    height_cm: float = Field(..., gt=100, lt=250, description="Height in cm")
    weight_kg: float = Field(..., gt=30,  lt=300, description="Weight in kg")

    @field_validator("height_cm", "weight_kg")
    @classmethod
    def round_one_decimal(cls, v: float) -> float:
        return round(v, 1)


class SessionResponse(BaseModel):
    """Safe public representation of a session — no internal URLs."""
    session_id:       str
    status:           SessionStatus
    gender:           Gender
    height_cm:        float
    weight_kg:        float
    has_face_photo:   bool
    has_body_photo:   bool
    has_avatar:       bool                    # true once avatar_url is stored
    avatar_status:    AvatarStatus            # pending | generating | ready | failed
    measurements:     Optional[Measurements] = None
    error_message:    Optional[str]          = None


# ─────────────────────────────────────────────────────────────────────────────
# Image upload
# ─────────────────────────────────────────────────────────────────────────────

class UploadFaceResponse(BaseModel):
    session_id:     str
    face_photo_url: str          # signed URL, safe to return to client
    status:         SessionStatus


# ─────────────────────────────────────────────────────────────────────────────
# Measurements extraction
# ─────────────────────────────────────────────────────────────────────────────
# Note: /session/extract accepts multipart/form-data (file + session cookie).
# No separate request body schema needed — FastAPI reads File(...) directly.

class ExtractResponse(BaseModel):
    session_id:   str
    status:       SessionStatus
    measurements: Measurements


# ─────────────────────────────────────────────────────────────────────────────
# Measurements update (user-edited values)
# ─────────────────────────────────────────────────────────────────────────────

class MeasurementUpdateRequest(BaseModel):
    """
    Partial or full update of the stored measurements.
    Only the four user-editable fields are accepted — bmi is always derived
    from height and weight and must never be overwritten by the client.
    All four fields are required: the frontend always sends the full set
    (edited or not), which avoids ambiguous partial-patch semantics.
    """
    chest_cm:          float = Field(..., gt=20, lt=200, description="Chest circumference in cm")
    waist_cm:          float = Field(..., gt=20, lt=200, description="Waist circumference in cm")
    hip_cm:            float = Field(..., gt=20, lt=200, description="Hip circumference in cm")
    shoulder_width_cm: float = Field(..., gt=15, lt=100, description="Shoulder width in cm")

    @field_validator("chest_cm", "waist_cm", "hip_cm", "shoulder_width_cm")
    @classmethod
    def round_one_decimal(cls, v: float) -> float:
        return round(v, 1)


class MeasurementUpdateResponse(BaseModel):
    session_id:   str
    measurements: Measurements    # echo back the full stored object (includes bmi)


# ─────────────────────────────────────────────────────────────────────────────
# Avatar generation
# ─────────────────────────────────────────────────────────────────────────────

class AvatarResponse(BaseModel):
    session_id:    str
    avatar_status: AvatarStatus
    avatar_url:    str           # signed URL of avatar image — safe to return to client
    provider:      str           # which service generated it: "gemini" | "fal" | "replicate"


# ─────────────────────────────────────────────────────────────────────────────
# Error
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error_code: str
    message:    str