import _isolate  # noqa: F401  (must come before any app import)
import os
import unittest
from unittest.mock import MagicMock, patch

from app.services import db


class DatabaseTimeoutTests(unittest.TestCase):
    def test_pool_has_bounded_defaults_and_overrides_without_connecting(self):
        cases = (
            ({"DB_CONNECT_TIMEOUT_SECONDS": "10", "DB_STATEMENT_TIMEOUT_MS": "30000", "DB_POOL_TIMEOUT_SECONDS": "10"},
             10, "-c statement_timeout=30000", 10),
            ({"DB_CONNECT_TIMEOUT_SECONDS": "3", "DB_STATEMENT_TIMEOUT_MS": "2500", "DB_POOL_TIMEOUT_SECONDS": "1.5"},
             3, "-c statement_timeout=2500", 1.5),
        )
        for env, connect, statement, acquisition in cases:
            with self.subTest(env=env), patch.dict(os.environ, {"SUPABASE_DB_URL": "mock-cloud", **env}), \
                    patch.object(db, "_pool", None), patch("psycopg_pool.ConnectionPool") as factory:
                self.assertIs(db.pool(), factory.return_value)
                options = factory.call_args.kwargs
                self.assertEqual(options["kwargs"]["connect_timeout"], connect)
                self.assertEqual(options["kwargs"]["options"], statement)
                self.assertEqual(options["timeout"], acquisition)
                self.assertTrue(options["kwargs"]["autocommit"])
                db.pool()
                factory.assert_called_once()

    def test_connection_acquisition_uses_the_configured_deadline(self):
        pool = MagicMock()
        with patch.dict(os.environ, {"DB_POOL_TIMEOUT_SECONDS": "0.5"}), patch.object(db, "pool", return_value=pool):
            with db.connection() as conn:
                self.assertIs(conn, pool.connection.return_value.__enter__.return_value)
        pool.connection.assert_called_once_with(timeout=0.5)

    def test_acquisition_timeout_propagates_without_an_unbounded_retry(self):
        from psycopg_pool import PoolTimeout
        pool = MagicMock()
        pool.connection.side_effect = PoolTimeout("busy")
        with patch.object(db, "pool", return_value=pool), self.assertRaises(PoolTimeout):
            with db.connection():
                self.fail("must not acquire a connection")
        pool.connection.assert_called_once()
