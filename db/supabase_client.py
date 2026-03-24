# db/supabase_client.py — Supabase client singleton
from functools import lru_cache
from supabase import create_client, Client
from config import get_settings


@lru_cache()
def get_supabase() -> Client:
    """
    Returns a cached Supabase client initialised with the secret key.

    We use the SECRET key (not anon key) because this backend is server-side
    and needs to bypass RLS for session management and storage uploads.
    Never expose this key to the frontend.
    """
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_secret_key)