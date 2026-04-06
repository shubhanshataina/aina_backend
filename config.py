# config.py — centralised settings loaded from .env
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv

# Explicitly load .env before pydantic-settings reads os.environ.
# Path is resolved relative to this file so it works regardless of
# which directory uvicorn is launched from.
ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(ENV_FILE, override=True)


class Settings(BaseSettings):
    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url:        str = Field(..., env="SUPABASE_URL")
    supabase_secret_key: str = Field(..., env="SUPABASE_SECRET_KEY")

    # ── Storage bucket names ──────────────────────────────────────────────────
    storage_bucket_face:   str = Field("aina-face-photos",   env="STORAGE_BUCKET_FACE")
    storage_bucket_body:   str = Field("aina-body-photos",   env="STORAGE_BUCKET_BODY")
    storage_bucket_avatar: str = Field("aina-avatars",       env="STORAGE_BUCKET_AVATAR")

    # ── GPU Worker ────────────────────────────────────────────────────────────
    worker_url: str = Field(..., env="WORKER_URL")

    # ── Avatar generation ─────────────────────────────────────────────────────
    # Supported values: "vertex" | "gemini"
    # "vertex"  → Gemini on Vertex AI via service account (recommended, uses GCP credits)
    # "gemini"  → Gemini via AI Studio API key (free tier, limited quota)
    avatar_provider: str = Field("vertex", env="AVATAR_PROVIDER")

    # Gemini AI Studio — only needed when AVATAR_PROVIDER=gemini
    gemini_api_key: str = Field("", env="GEMINI_API_KEY")

    # ── Vertex AI ─────────────────────────────────────────────────────────────
    # Required when AVATAR_PROVIDER=vertex.
    #
    # VERTEX_PROJECT_ID       — GCP project ID (e.g. "my-project-123456")
    #                           Found in the GCP Console header dropdown.
    #                           NOT the project number — the string ID.
    #
    # VERTEX_LOCATION         — Must be "us-central1". Gemini image generation
    #                           is only available in us-central1 right now.
    #
    # GOOGLE_CREDENTIALS_JSON — The entire contents of the service account
    #                           JSON key file as a single-line string.
    #                           In production: add as a Secret env var on your
    #                           hosting platform (Railway / Render / Fly.io).
    #                           Locally: paste the JSON into .env (see below).
    #                           Required SA role: roles/aiplatform.user
    #
    # How to get the JSON for .env locally:
    #   Mac/Linux: cat your-key.json | tr -d '\n'  → paste the output as the value
    #   Or just paste the raw JSON — dotenv handles multi-line values in double quotes:
    #     GOOGLE_CREDENTIALS_JSON='{ "type": "service_account", ... }'
    vertex_project_id:        str = Field("", env="VERTEX_PROJECT_ID")
    vertex_location:          str = Field("us-central1", env="VERTEX_LOCATION")
    google_credentials_json:  str = Field("", env="GOOGLE_CREDENTIALS_JSON")

    # ── Session cookie ────────────────────────────────────────────────────────
    session_cookie_name:     str       = Field("aina_session",       env="SESSION_COOKIE_NAME")
    session_cookie_max_age:  int       = Field(86400,                env="SESSION_COOKIE_MAX_AGE")
    session_cookie_secure:   bool      = Field(True,                 env="SESSION_COOKIE_SECURE")
    session_cookie_samesite: str       = Field("none",               env="SESSION_COOKIE_SAMESITE")
    # Domain must start with a leading dot so the cookie is valid on all subdomains.
    # Example: ".ragerstudios.com" covers both ainaapi.ragerstudios.com (sets it)
    # and aina3d.ragerstudios.com (reads it).
    # Leave blank in local dev — the browser binds to localhost automatically.
    session_cookie_domain:   str | None = Field(None,                env="SESSION_COOKIE_DOMAIN")

    # ── CORS ──────────────────────────────────────────────────────────────────
    # In pydantic-settings v2, field names map directly to env var names
    # (uppercased). So this field reads ALLOWED_ORIGINS from the environment.
    allowed_origins: str = Field("http://localhost:3000", alias="ALLOWED_ORIGINS")

    @property
    def allowed_origins_list(self) -> list[str]:
        origins = []
        for raw in self.allowed_origins.split(","):
            raw = raw.strip()
            if not raw:
                continue
            parsed = urlparse(raw)
            origin = f"{parsed.scheme}://{parsed.netloc}"
            if origin not in origins:
                origins.append(origin)
        return origins

    # ── App ───────────────────────────────────────────────────────────────────
    app_env:   str = Field("development", env="APP_ENV")
    log_level: str = Field("INFO",        env="LOG_LEVEL")

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    model_config = {
        "env_file":          str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra":             "ignore",
        "populate_by_name":  True,   # allow access by field name AND alias
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached singleton — import and call this everywhere."""
    return Settings()