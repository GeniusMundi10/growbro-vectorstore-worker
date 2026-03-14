from supabase import create_client, Client
import os

# ==================== GrowBro (Primary) ====================
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase credentials not set in environment variables.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==================== GroChurch (Secondary) ====================
GROCHURCH_SUPABASE_URL = os.environ.get("GROCHURCH_SUPABASE_URL")
GROCHURCH_SUPABASE_KEY = os.environ.get("GROCHURCH_SUPABASE_KEY")

grochurch_supabase: Client | None = None
if GROCHURCH_SUPABASE_URL and GROCHURCH_SUPABASE_KEY:
    grochurch_supabase = create_client(GROCHURCH_SUPABASE_URL, GROCHURCH_SUPABASE_KEY)
    print("[Supabase] GroChurch client initialized.")
else:
    print("[Supabase] GroChurch credentials not set. GroChurch routing disabled.")
