"""Serve a Vite build, with API routes excluded from the SPA fallback."""

from fastapi import FastAPI
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles


class FrontendFiles(StaticFiles):
    async def get_response(self, path, scope):
        route = path.replace("\\", "/").lstrip("/")
        if route == "api" or route.startswith("api/"):
            raise HTTPException(404)
        try:
            return await super().get_response(path, scope)
        except HTTPException as error:
            if error.status_code != 404:
                raise
            return await super().get_response("index.html", scope)


def mount_frontend(app: FastAPI, directory: str) -> None:
    # Mount last: known API routes take priority over the catch-all static mount.
    app.mount("/", FrontendFiles(directory=directory, html=True), name="frontend")
