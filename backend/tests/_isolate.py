"""Imported first by every test module: blank the cloud settings so app.main's load_dotenv (which
never overrides a set variable) cannot point the tests at the real Supabase / R2 from ../.env."""

import os

for name in ("SUPABASE_DB_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY",
             "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ[name] = ""
