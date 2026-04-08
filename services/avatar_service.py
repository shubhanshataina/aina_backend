# services/avatar_service.py
#
# Provider-agnostic avatar generation service.
#
# ── Provider map ─────────────────────────────────────────────────────────────
# "vertex"  → Gemini on Vertex AI via google-cloud-aiplatform SDK (primary)
#             The SDK initialises once from GOOGLE_CREDENTIALS_JSON and handles
#             all OAuth2 token fetching, expiry, and refresh automatically.
#             Uses GCP project credits ($300 trial covers ~500k avatars).
#
# "gemini"  → Gemini via AI Studio REST API (legacy fallback)
#             Requires GEMINI_API_KEY with billing enabled.
#
# ── Why the SDK and not raw httpx for Vertex ─────────────────────────────────
# Raw httpx requires manually fetching an OAuth2 bearer token, managing its
# expiry, and refreshing it — all complexity that google-cloud-aiplatform
# handles transparently. The SDK's GenerativeModel.generate_content() is a
# single call that does everything: auth, request, response parsing.
# The only thing we do ourselves is extract the image bytes from the response,
# since the SDK returns a GenerateContentResponse object, not raw bytes.
#
# ── Why Gemini on Vertex, not Imagen 3 ───────────────────────────────────────
# Imagen 3 (imagen-3.0-generate-002) is text → image only. It cannot accept a
# reference photo as input, so it cannot produce an avatar that looks like the
# actual user. Gemini multimodal accepts image + text in and returns an image
# out — which is the only model capability that serves avatar generation.
#
# ── SDK client singleton ──────────────────────────────────────────────────────
# vertexai.init() is called once at first use and cached. Subsequent calls to
# GenerativeModel() reuse the same initialised SDK context. The SDK manages the
# underlying credentials object (parsed from GOOGLE_CREDENTIALS_JSON) and
# refreshes the bearer token automatically when it expires.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading

import aiohttp
from fastapi import HTTPException

from config import get_settings

log = logging.getLogger("app")


# ─────────────────────────────────────────────────────────────────────────────
# VERTEX AI CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# Gemini model on Vertex AI — multimodal, image-in + image-out.
# Update this string when a newer stable model is available in the Model Garden.
_VERTEX_MODEL = "gemini-2.5-flash-image"

# Timeout for the legacy Gemini AI Studio path.
# The Vertex SDK manages its own internal timeouts.
_GEMINI_TIMEOUT = aiohttp.ClientTimeout(total=160.0, connect=10.0, sock_read=150.0)


# ─────────────────────────────────────────────────────────────────────────────
# SDK SINGLETON
# ─────────────────────────────────────────────────────────────────────────────
# vertexai.init() is the one-time setup call that tells the SDK which project,
# region, and credentials to use for all subsequent API calls in this process.
# It is cheap to call repeatedly (it's just setting module-level state), but
# we guard it with a flag + lock anyway to make intent explicit and avoid any
# edge-case double-init behaviour on concurrent first requests.

_vertex_initialised = False
_vertex_init_lock   = threading.Lock()

# ── Vertex generation concurrency guard ───────────────────────────────────────
# Allows only one generate_content() call at a time across all concurrent
# requests in this process. Vertex AI image generation models have tight
# per-minute quotas. Two simultaneous requests (e.g. from a frontend retry
# racing with a still-in-flight first request) both passing the DB-level 409
# guard and hitting Vertex concurrently causes quota exhaustion (finish_reason=11)
# on the first call and a hard 429 on the second.
#
# asyncio.Semaphore(1) is sufficient because uvicorn runs with workers=1.
# It is process-local — if you ever move to multiple workers, replace this
# with a distributed lock (e.g. Supabase advisory lock or Redis SETNX).
#
# The semaphore is created lazily on first use (not at import time) because
# it must be created inside a running event loop.
_vertex_semaphore: asyncio.Semaphore | None = None
_vertex_semaphore_lock = threading.Lock()

def _get_vertex_semaphore() -> asyncio.Semaphore:
    """Return the process-level Vertex generation semaphore, creating it if needed."""
    global _vertex_semaphore
    if _vertex_semaphore is None:
        with _vertex_semaphore_lock:
            if _vertex_semaphore is None:
                _vertex_semaphore = asyncio.Semaphore(1)
    return _vertex_semaphore

def _bake_orientation(image_bytes: bytes) -> bytes:
    """
    Rotate pixel data to match EXIF Orientation tag, then strip all EXIF.

    Vertex AI returns JPEGs with an Orientation tag (commonly value 6 = 90°CW).
    Chrome and Android apply the tag visually. Safari on iOS does not — it
    renders raw pixels, making a portrait image appear in landscape.

    PIL's ImageOps.exif_transpose() bakes the rotation into actual pixels
    and clears the tag so every client sees the same result with no CSS needed.
    """
    from PIL import Image, ImageOps
    import io

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)   # rotates pixels + clears Orientation tag
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=95)
    return out.getvalue()

def _ensure_vertex_initialised() -> None:
    """
    Initialise the Vertex AI SDK from GOOGLE_CREDENTIALS_JSON.

    Called explicitly from the FastAPI lifespan at startup (via
    warmup_vertex_sdk), so this is always a no-op by the time any
    request arrives. The lazy guard is kept as a safety net only.
    """
    global _vertex_initialised

    if _vertex_initialised:
        return

    with _vertex_init_lock:
        if _vertex_initialised:
            return

        # Import here so the module loads cleanly on machines without the SDK
        # installed, as long as AVATAR_PROVIDER != "vertex"
        try:
            import vertexai
            from google.oauth2 import service_account
        except ImportError:
            raise RuntimeError(
                "Vertex AI SDK is not installed.\n"
                "Run: pip install google-cloud-aiplatform"
            )

        settings = get_settings()

        if not settings.google_credentials_json:
            raise RuntimeError(
                "GOOGLE_CREDENTIALS_JSON is not set.\n"
                "Local:      paste the service account JSON into .env\n"
                "Production: add it as a Secret environment variable on your host"
            )

        if not settings.vertex_project_id:
            raise RuntimeError(
                "VERTEX_PROJECT_ID is not set in .env.\n"
                "Set it to your GCP project ID (the string, not the number)."
            )

        # Parse and validate the credentials JSON
        try:
            key_dict = json.loads(settings.google_credentials_json)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"GOOGLE_CREDENTIALS_JSON is not valid JSON: {e}\n"
                "The value should be the full contents of the service account "
                "key file, not a file path."
            )

        if key_dict.get("type") != "service_account":
            raise RuntimeError(
                "GOOGLE_CREDENTIALS_JSON does not contain a service account key. "
                f"Got type={key_dict.get('type')!r}. "
                "Download a JSON key from GCP → IAM & Admin → Service Accounts."
            )

        # Build the credentials object and hand it to the SDK.
        # From this point the SDK owns auth entirely — it fetches the initial
        # bearer token and refreshes it before expiry with no action needed here.
        try:
            credentials = service_account.Credentials.from_service_account_info(
                key_dict,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to build credentials from JSON: {e}")

        vertexai.init(
            project     = settings.vertex_project_id,
            location    = settings.vertex_location,
            credentials = credentials,
        )

        _vertex_initialised = True
        log.info(
            f"Vertex AI SDK initialised — "
            f"project={settings.vertex_project_id} "
            f"location={settings.vertex_location} "
            f"model={_VERTEX_MODEL} "
            f"sa={key_dict.get('client_email')}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP WARMUP
# ─────────────────────────────────────────────────────────────────────────────

async def warmup_vertex_sdk() -> None:
    """
    Call this from the FastAPI lifespan at startup.

    Does two things that eliminate the first-request timeout:

    1. Runs _ensure_vertex_initialised() — parses credentials, calls
       vertexai.init(), sets the module-level flag.

    2. Instantiates GenerativeModel() once in a thread. This triggers the
       underlying gRPC channel establishment to Vertex AI (us-central1).
       That channel open is the hidden latency — it can take 10–20 seconds
       on a cold process. Doing it at startup means it is fully warm before
       any user request arrives, and generate_content() sees a ~0ms channel
       setup cost instead of paying it inside the 90s request window.

    Non-fatal: if credentials are missing or Vertex is unreachable at startup
    we log a warning but let the server boot. The first real request will then
    hit _ensure_vertex_initialised() via the lazy guard and fail with a clear
    VERTEX_NOT_CONFIGURED error rather than silently timing out.
    """
    settings = get_settings()
    if settings.avatar_provider.lower().strip() != "vertex":
        return   # nothing to warm up for the Gemini AI Studio path

    try:
        _ensure_vertex_initialised()
    except RuntimeError as e:
        log.warning(f"Vertex SDK init skipped at startup (will retry on first request): {e}")
        return

    # Instantiate GenerativeModel in a thread — this opens the gRPC channel.
    # We do not make an actual generate_content() call (that costs quota);
    # channel establishment happens on model instantiation alone.
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            _warmup_grpc_channel,
        )
        log.info("Vertex AI gRPC channel warmed up — first avatar request will not pay init cost")
    except Exception as e:
        log.warning(f"Vertex gRPC warmup failed (non-fatal): {e}")


def _warmup_grpc_channel() -> None:
    """
    Instantiate GenerativeModel to force gRPC channel establishment.
    Called once from warmup_vertex_sdk() via run_in_executor at startup.
    """
    from vertexai.generative_models import GenerativeModel
    GenerativeModel(_VERTEX_MODEL)
    log.info("Vertex AI GenerativeModel instantiated — gRPC channel open")


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_CLOTHING = {
    "male":   "fitted black compression t-shirt and black compression shorts",
    "female": "fitted black bodycon tank yoga jumpsuit",
}

def _build_prompt(gender: str) -> str:
    clothing = _CLOTHING.get(gender.lower(), _CLOTHING["male"])
    return (
        f"Generate a photorealistic full-body 2D digital avatar of the person "
        f"in this image. Follow these requirements exactly:\n"
        f"- Preserve the person's exact face, skin tone, hair, and body "
        f"proportions — do NOT alter body shape or weight in any way\n"
        f"- Dress them in {clothing}\n"
        f"- A-pose: arms slightly away from body, palms facing forward, "
        f"feet shoulder-width apart\n"
        f"- Clean white studio background with soft, even professional lighting\n"
        f"- Full body visible from top of head to feet\n"
        f"- Sharp, high quality, fashion photography style\n"
        f"- Do NOT add accessories, jewellery, watches, or footwear\n"
        f"- Do NOT change facial features or expression\n"
        f"- Output a single portrait-oriented image"
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

async def generate_avatar(
    session_id:       str,
    body_image_bytes: bytes,
    body_image_mime:  str,
    gender:           str,
) -> tuple[bytes, str]:
    """
    Generate a photorealistic avatar from the user's body photo.

    Returns:
        (avatar_bytes, provider_name)
        avatar_bytes  — raw JPEG bytes of the generated avatar
        provider_name — "vertex" | "gemini"

    Raises:
        HTTPException with sanitised error_code + message on any failure.
    """
    settings = get_settings()
    provider = settings.avatar_provider.lower().strip()
    prompt   = _build_prompt(gender)

    log.info(f"[{session_id}] Avatar generation — provider={provider} gender={gender}")

    if provider == "vertex":
        return await _generate_via_vertex(
            session_id, body_image_bytes, body_image_mime, prompt
        )

    if provider == "gemini":
        return await _generate_via_gemini(
            session_id, body_image_bytes, body_image_mime, prompt
        )

    raise HTTPException(status_code=500, detail={
        "error_code": "UNKNOWN_AVATAR_PROVIDER",
        "message":    f"Avatar provider '{provider}' is not configured.",
    })


# ─────────────────────────────────────────────────────────────────────────────
# VERTEX AI IMPLEMENTATION — SDK path
# ─────────────────────────────────────────────────────────────────────────────

def _call_vertex_sync(
    body_image_bytes: bytes,
    body_image_mime:  str,
    prompt:           str,
) -> bytes:
    """
    Synchronous wrapper around the Vertex AI SDK call.
    Must be run via run_in_executor — the SDK's generate_content() is blocking.

    Why blocking? The google-cloud-aiplatform SDK uses grpc under the hood,
    which is synchronous in its Python implementation. It does not expose an
    async interface. Rather than adding grpc-asyncio as a dependency just to
    avoid the executor, we run the SDK call in a thread pool — which is the
    standard pattern for any blocking I/O in an async FastAPI application.

    Retry logic: gemini-2.5-flash-image (preview) has a documented first-call
    behaviour where generate_content() occasionally returns a text-only response
    (the model narrates what it would generate rather than generating it) on the
    first request after process start, even with a warm gRPC channel. This is a
    model-level quirk, not a network error. One automatic retry is sufficient —
    the second call consistently returns the image. The retry is handled here,
    inside the sync function, so the async caller never needs to know about it.
    """
    from vertexai.generative_models import GenerativeModel, GenerationConfig, Part, Image

    model      = GenerativeModel(_VERTEX_MODEL)
    image_part = Part.from_image(Image.from_bytes(body_image_bytes))

    # GenerationConfig object is required — passing a raw dict causes some SDK
    # versions to silently drop unrecognised fields, losing the IMAGE modality
    # constraint and defaulting to text-only output.
    gen_config = GenerationConfig(response_modalities=["IMAGE", "TEXT"])

    # finish_reason int values returned by the Vertex SDK.
    # The SDK may not recognise newer enum values and falls back to the raw int,
    # so we check both the .name string and the raw int value.
    #
    # 1 = STOP (normal completion)
    # 2 = MAX_TOKENS
    # 3 = SAFETY
    # 8 = BLOCKLIST
    # 9 = PROHIBITED_CONTENT
    # 11 = IMAGE_GENERATION_QUOTA_EXCEEDED  ← what we are seeing in logs
    _QUOTA_FINISH_REASONS     = {"IMAGE_GENERATION_QUOTA_EXCEEDED", "RESOURCE_EXHAUSTED"}
    _QUOTA_FINISH_REASON_INTS = {11}
    _SAFETY_FINISH_REASONS    = {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"}
    _SAFETY_FINISH_REASON_INTS = {3, 8, 9}

    def _attempt() -> bytes | None:
        """
        Single generate_content attempt.
        Returns image bytes if found, None if the response contained only text
        and the finish reason is a transient/retryable model quirk.

        Raises:
          _SafetyBlockedError  — finish reason is safety/content block (not retriable)
          _QuotaExceededError  — finish reason is quota exhaustion (not retriable, do not retry)
        """
        response  = model.generate_content(
            [image_part, prompt],
            generation_config=gen_config,
        )
        candidate = response.candidates[0]

        for part in candidate.content.parts:
            if part.inline_data and part.inline_data.data:
                return part.inline_data.data   # raw bytes, not base64

        # No image in the response — inspect finish_reason before deciding to retry.
        finish_reason      = candidate.finish_reason
        finish_reason_name = getattr(finish_reason, "name", str(finish_reason))
        finish_reason_int  = finish_reason if isinstance(finish_reason, int) else getattr(finish_reason, "value", None)

        # Quota exhaustion — do NOT retry, raise immediately.
        # finish_reason=11 (IMAGE_GENERATION_QUOTA_EXCEEDED) means a second call
        # will also fail and wastes quota budget.
        if finish_reason_name in _QUOTA_FINISH_REASONS or finish_reason_int in _QUOTA_FINISH_REASON_INTS:
            log.warning(
                f"Vertex quota finish reason detected — not retrying. "
                f"finish_reason={finish_reason_name}({finish_reason_int})"
            )
            raise _QuotaExceededError(f"finish_reason={finish_reason_name}")

        # Safety / content block — not retriable.
        if finish_reason_name in _SAFETY_FINISH_REASONS or finish_reason_int in _SAFETY_FINISH_REASON_INTS:
            raise _SafetyBlockedError()

        # Anything else with no image — log and treat as transiently retryable.
        text_preview = " | ".join(
            p.text[:120] for p in candidate.content.parts if hasattr(p, "text") and p.text
        ) or "<no text>"
        log.warning(
            f"Vertex returned no image part — will retry once. "
            f"finish_reason={finish_reason_name}({finish_reason_int}) "
            f"text_preview={text_preview!r}"
        )
        return None

    # First attempt
    result = _attempt()
    if result is not None:
        return result

    # One retry — text-only first response on preview models is a known transient quirk.
    # We only reach here if finish_reason was NOT quota or safety (those raise above).
    log.info("Retrying Vertex generate_content after no-image first response")
    result = _attempt()
    if result is not None:
        return result

    raise _NoImageError("Both attempts returned no image part")


class _SafetyBlockedError(Exception):
    pass

class _NoImageError(Exception):
    pass

class _QuotaExceededError(Exception):
    """
    Raised when Vertex returns finish_reason=IMAGE_GENERATION_QUOTA_EXCEEDED (11).
    Caught in _generate_via_vertex and mapped to AVATAR_RATE_LIMITED.
    Must NOT be retried — the quota is exhausted for this project/period.
    """
    pass


async def _generate_via_vertex(
    session_id:       str,
    body_image_bytes: bytes,
    body_image_mime:  str,
    prompt:           str,
) -> tuple[bytes, str]:
    """
    Generate an avatar via the Vertex AI SDK.

    The SDK call is blocking so we offload it to a thread pool executor.
    The event loop is free while the SDK waits for Vertex AI to respond
    (typically 15–45 seconds for image generation).
    """
    # Ensure SDK is initialised — fast no-op after first call
    try:
        _ensure_vertex_initialised()
    except RuntimeError as e:
        log.error(f"[{session_id}] Vertex SDK init failed: {e}")
        raise HTTPException(status_code=500, detail={
            "error_code": "VERTEX_NOT_CONFIGURED",
            "message":    "Avatar service is not configured. Please contact support.",
        })

    log.info(f"[{session_id}] Calling Vertex AI SDK — model={_VERTEX_MODEL}")

    semaphore = _get_vertex_semaphore()
    queue_pos = semaphore._value  # 0 = will queue, 1 = will run immediately
    if queue_pos == 0:
        log.info(f"[{session_id}] Vertex semaphore busy — queuing (prevents concurrent quota hit)")

    try:
        async with semaphore:
            log.info(f"[{session_id}] Vertex semaphore acquired — starting generate_content()")
            avatar_bytes = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _call_vertex_sync(body_image_bytes, body_image_mime, prompt),
            )

            # ── Fix EXIF orientation (Safari iOS bug) ─────────────────────
            # Vertex embeds an EXIF Orientation tag. Chrome/Android apply it
            # visually; Safari renders raw pixels — portrait becomes landscape.
            # _bake_orientation() rotates pixels to match the tag, then strips
            # all EXIF so every browser sees identical output.
            avatar_bytes = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: _bake_orientation(avatar_bytes),
            )
    except _QuotaExceededError as e:
        # finish_reason=11 — quota exhausted at the model/project level.
        # Raised from _attempt() without a retry so we don't double-spend quota.
        # The outer exception handler also catches SDK-level 429s (resource exhausted)
        # that arrive as gRPC exceptions rather than finish_reason values.
        log.warning(f"[{session_id}] Vertex quota finish reason: {e}")
        raise HTTPException(status_code=429, detail={
            "error_code": "AVATAR_RATE_LIMITED",
            "message":    "Avatar generation quota reached. Please try again in a few minutes.",
        })
    except _SafetyBlockedError:
        log.warning(f"[{session_id}] Vertex safety filter blocked avatar generation")
        raise HTTPException(status_code=422, detail={
            "error_code": "AVATAR_SAFETY_BLOCKED",
            "message":    (
                "Avatar generation was blocked. Please ensure your photo "
                "shows a clear, neutral standing pose."
            ),
        })
    except _NoImageError as e:
        log.error(
            f"[{session_id}] Vertex returned no image after retry: {e}. "
            f"This should be rare — if it recurs check the model endpoint and "
            f"response_modalities config."
        )
        raise HTTPException(status_code=500, detail={
            "error_code": "AVATAR_PARSE_FAILED",
            "message":    "Avatar generation failed. Please try again.",
        })
    except Exception as e:
        # Map SDK exceptions to client-safe errors
        err_str = str(e).lower()

        if "deadline exceeded" in err_str or "timeout" in err_str:
            log.error(f"[{session_id}] Vertex timeout: {e}")
            raise HTTPException(status_code=504, detail={
                "error_code": "AVATAR_TIMEOUT",
                "message":    "Avatar generation is taking longer than expected. Please try again.",
            })

        if "permission denied" in err_str or "unauthenticated" in err_str:
            log.error(
                f"[{session_id}] Vertex auth error: {e}\n"
                f"Checklist:\n"
                f"  1. Vertex AI API enabled at console.cloud.google.com/apis?\n"
                f"  2. Service account has roles/aiplatform.user?\n"
                f"  3. VERTEX_PROJECT_ID matches the project in GOOGLE_CREDENTIALS_JSON?\n"
                f"  4. VERTEX_LOCATION=us-central1? (image gen only in us-central1)"
            )
            raise HTTPException(status_code=500, detail={
                "error_code": "VERTEX_AUTH_ERROR",
                "message":    "Avatar service configuration error. Please contact support.",
            })

        if "not found" in err_str or "does not exist" in err_str:
            log.error(
                f"[{session_id}] Vertex model not found: {e}\n"
                f"Current model: {_VERTEX_MODEL} — check Vertex AI Model Garden."
            )
            raise HTTPException(status_code=500, detail={
                "error_code": "VERTEX_MODEL_NOT_FOUND",
                "message":    "Avatar service configuration error. Please contact support.",
            })

        if "resource exhausted" in err_str or "quota" in err_str:
            log.warning(f"[{session_id}] Vertex quota exceeded: {e}")
            raise HTTPException(status_code=429, detail={
                "error_code": "AVATAR_RATE_LIMITED",
                "message":    "Avatar service is busy. Please wait a moment and try again.",
            })

        log.error(f"[{session_id}] Vertex unexpected error: {e}")
        raise HTTPException(status_code=500, detail={
            "error_code": "AVATAR_FAILED",
            "message":    "Avatar generation failed. Please try again.",
        })

    log.info(f"[{session_id}] Vertex avatar generated — {len(avatar_bytes):,} bytes")
    return avatar_bytes, "vertex"


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY: GEMINI AI STUDIO — raw httpx, no SDK
# Set AVATAR_PROVIDER=gemini to use. Requires a paid billing plan.
# ─────────────────────────────────────────────────────────────────────────────

_GEMINI_MODEL    = "gemini-2.0-flash-exp"
_GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent?key={key}"
)

async def _generate_via_gemini(
    session_id:       str,
    body_image_bytes: bytes,
    body_image_mime:  str,
    prompt:           str,
) -> tuple[bytes, str]:
    settings = get_settings()

    if not settings.gemini_api_key:
        raise HTTPException(status_code=500, detail={
            "error_code": "GEMINI_NOT_CONFIGURED",
            "message":    "Avatar service is not configured. Please try again later.",
        })

    image_b64 = base64.standard_b64encode(body_image_bytes).decode("utf-8")
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": body_image_mime, "data": image_b64}},
                {"text": prompt},
            ]
        }],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
    }
    url = _GEMINI_ENDPOINT.format(model=_GEMINI_MODEL, key=settings.gemini_api_key)

    try:
        async with aiohttp.ClientSession(timeout=_GEMINI_TIMEOUT) as client:
            async with client.post(url, json=payload) as response:
                response_status = response.status
                response_text   = await response.text()
    except aiohttp.ServerTimeoutError:
        raise HTTPException(status_code=504, detail={
            "error_code": "AVATAR_TIMEOUT",
            "message":    "Avatar generation is taking longer than expected. Please try again.",
        })
    except Exception as e:
        log.error(f"[{session_id}] Gemini connection error: {e}")
        raise HTTPException(status_code=503, detail={
            "error_code": "AVATAR_SERVICE_UNAVAILABLE",
            "message":    "Avatar service is temporarily unavailable. Please try again.",
        })

    if response_status != 200:
        import json as _json
        try:
            msg = _json.loads(response_text).get("error", {}).get("message", "")
        except Exception:
            msg = response_text[:200]
        log.error(f"[{session_id}] Gemini error {response_status}: {msg}")
        raise HTTPException(status_code=500, detail={
            "error_code": "AVATAR_FAILED",
            "message":    "Avatar generation failed. Please try again.",
        })

    import json as _json
    try:
        data         = _json.loads(response_text)
        candidates   = data.get("candidates", [])
        parts        = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            inline = part.get("inline_data") or part.get("inlineData")
            if inline and inline.get("data"):
                avatar_bytes = base64.b64decode(inline["data"])
                log.info(f"[{session_id}] Gemini avatar generated — {len(avatar_bytes):,} bytes")
                return avatar_bytes, "gemini"
        raise ValueError("No image part in response")
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"[{session_id}] Gemini response parse failed: {e}")
        raise HTTPException(status_code=500, detail={
            "error_code": "AVATAR_PARSE_FAILED",
            "message":    "Avatar generation failed. Please try again.",
        })


# ─────────────────────────────────────────────────────────────────────────────
# FAL.AI STUB
# ─────────────────────────────────────────────────────────────────────────────

# async def _generate_via_fal(...) -> tuple[bytes, str]:
#     """
#     Set AVATAR_PROVIDER=fal and FAL_API_KEY in .env.
#     Better face consistency than Gemini for avatar use cases.
#     Docs: https://fal.ai/models/fal-ai/instant-id
#     """
#     pass