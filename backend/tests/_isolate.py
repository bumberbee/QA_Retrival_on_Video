"""Imported first by every test module: never read .env or use real cloud settings."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The installed dotenv version does not honor PYTHON_DOTENV_DISABLED. Patch before
# app.main imports it so even local backend tests cannot open a credential file.
_dotenv_patch = patch("dotenv.load_dotenv", return_value=False)
_dotenv_patch.start()

for name in ("SUPABASE_DB_URL", "SUPABASE_URL", "SUPABASE_ANON_KEY",
             "R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
    os.environ[name] = ""
