# services/correction_service.py
#
# Writes to the `measurement_corrections` table after every PATCH /session/measurements.
# This table is the calibration dataset for hmr_worker.py — see the SQL migration
# for the full schema rationale and reference queries.
#
# DESIGN NOTES
# ────────────
# • This service is append-only. Corrections are immutable events; we never
#   update or delete rows here. If a user retakes their photo and re-confirms,
#   a second row is inserted. Both rows are valid calibration data.
#
# • Insertion is non-fatal to the PATCH route. If the DB write fails (transient
#   Supabase error, schema mismatch after a migration), the user's measurement
#   update still succeeds. We log the failure loudly so it shows up in monitoring,
#   but we never surface it as an API error. The calibration dataset missing one
#   row is far less bad than the user seeing an error on confirm.
#
# • raw_* columns (HMR2 mesh values before the anthropometric blend) are read
#   from the session row's `raw_measurements` column if it exists. They are
#   nullable — early sessions won't have them, and that's fine for calibration
#   (the ai_* values still tell you the final output error).

import logging
from datetime import datetime, timezone

from db.supabase_client import get_supabase

log = logging.getLogger("app")

TABLE = "measurement_corrections"


def _bmi_bracket(bmi: float) -> str:
    if bmi < 18.5:  return "underweight"
    if bmi < 25.0:  return "normal"
    if bmi < 30.0:  return "overweight"
    return "obese"


def record_correction(
    session_id:      str,
    session_row:     dict,
    ai_measurements: dict,
    user_patch:      dict,
) -> None:
    """
    Insert one row into measurement_corrections.

    Called from PATCH /session/measurements in main.py, after the session
    measurements have been successfully updated.

    Args:
        session_id      — the session UUID (for logging and the FK)
        session_row     — the raw session DB row, used for context fields
                          (gender, height_cm, weight_kg, bmi, blend methods)
        ai_measurements — the measurements dict as it existed in the DB
                          *before* the user's edit (the AI output we're scoring)
        user_patch      — the four values the user submitted
                          {chest_cm, waist_cm, hip_cm, shoulder_width_cm}

    This function is non-fatal: any exception is caught, logged, and swallowed.
    """
    try:
        bmi = float(ai_measurements.get("bmi", 0))

        # raw_measurements holds the pre-blend HMR2 mesh values if the session
        # was processed after we started capturing them. Nullable — fine.
        raw = session_row.get("raw_measurements") or {}

        row = {
            # Session context
            "session_id":   session_id,
            "gender":       session_row["gender"],
            "height_cm":    session_row["height_cm"],
            "weight_kg":    session_row["weight_kg"],
            "bmi":          round(bmi, 1),
            "bmi_bracket":  _bmi_bracket(bmi),

            # Raw HMR2 mesh values (pre-blend) — nullable
            "raw_chest_cm":    raw.get("chest_cm"),
            "raw_waist_cm":    raw.get("waist_cm"),
            "raw_hip_cm":      raw.get("hip_cm"),
            "raw_shoulder_cm": raw.get("shoulder_width_cm"),

            # Final AI output (what the user saw and may have corrected)
            "ai_chest_cm":    ai_measurements["chest_cm"],
            "ai_waist_cm":    ai_measurements["waist_cm"],
            "ai_hip_cm":      ai_measurements["hip_cm"],
            "ai_shoulder_cm": ai_measurements["shoulder_width_cm"],

            # User-submitted values
            "user_chest_cm":    user_patch["chest_cm"],
            "user_waist_cm":    user_patch["waist_cm"],
            "user_hip_cm":      user_patch["hip_cm"],
            "user_shoulder_cm": user_patch["shoulder_width_cm"],

            # Blend method tags stored on the session row (captured at extraction time)
            "body_method":     session_row.get("body_method"),
            "shoulder_method": session_row.get("shoulder_method"),

            "corrected_at": datetime.now(timezone.utc).isoformat(),
        }

        get_supabase().table(TABLE).insert(row).execute()

        # Structured log for immediate visibility in app.log / monitoring
        _log_correction(session_id, ai_measurements, user_patch, bmi, session_row["gender"])

    except Exception as exc:
        # Non-fatal: correction logging must never break the user flow.
        log.error(
            f"[{session_id}] Failed to write measurement correction to DB: {exc}. "
            f"This is a calibration data loss — investigate but do not surface to user."
        )


def _log_correction(
    session_id:      str,
    ai:              dict,
    user:            dict,
    bmi:             float,
    gender:          str,
) -> None:
    """
    Emit a structured log line summarising the correction.

    Format is parseable by log aggregators (Datadog, Loki, etc.) and also
    human-readable in app.log. Each field is key=value so you can grep or
    filter without a SQL query when iterating quickly.

    Fields:
        event         — always "measurement_correction" for easy filtering
        has_edit      — true if the user changed at least one value
        *_delta       — signed difference (user - ai); positive = AI underestimated
        *_pct_error   — |delta| / ai × 100, so you can spot proportional errors
    """
    deltas = {
        "chest":    round(user["chest_cm"]          - ai["chest_cm"],          1),
        "waist":    round(user["waist_cm"]          - ai["waist_cm"],          1),
        "hip":      round(user["hip_cm"]            - ai["hip_cm"],            1),
        "shoulder": round(user["shoulder_width_cm"] - ai["shoulder_width_cm"], 1),
    }

    def pct(delta, base):
        return round(abs(delta) / base * 100, 1) if base else 0.0

    has_edit = any(d != 0.0 for d in deltas.values())

    log.info(
        f"[{session_id}] event=measurement_correction "
        f"gender={gender} bmi={bmi:.1f} bmi_bracket={_bmi_bracket(bmi)} "
        f"has_edit={has_edit} "
        f"chest_ai={ai['chest_cm']} chest_user={user['chest_cm']} "
        f"chest_delta={deltas['chest']} chest_pct_error={pct(deltas['chest'], ai['chest_cm'])} "
        f"waist_ai={ai['waist_cm']} waist_user={user['waist_cm']} "
        f"waist_delta={deltas['waist']} waist_pct_error={pct(deltas['waist'], ai['waist_cm'])} "
        f"hip_ai={ai['hip_cm']} hip_user={user['hip_cm']} "
        f"hip_delta={deltas['hip']} hip_pct_error={pct(deltas['hip'], ai['hip_cm'])} "
        f"shoulder_ai={ai['shoulder_width_cm']} shoulder_user={user['shoulder_width_cm']} "
        f"shoulder_delta={deltas['shoulder']} shoulder_pct_error={pct(deltas['shoulder'], ai['shoulder_width_cm'])}"
    )