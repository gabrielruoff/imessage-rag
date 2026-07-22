"""FastAPI app assembly: CORS, routers, health, DB init, and the optional
production static mount. Run with:

    uvicorn api.main:app --host $API_HOST --port $API_PORT

(single-user, local only -- see api/README.md).
"""

import contextlib
import socket
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import config

from . import store
from .chat_stream import router as chat_router
from .conversations import router as conversations_router
from .models import Health


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure app.db exists before the first request.
    store.init_db()
    yield


app = FastAPI(title="imessage-rag API", lifespan=lifespan)

# Local dev frontend (Vite) lives on a different origin, so it needs CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ollama_reachable(timeout=0.5) -> bool:
    """Best-effort TCP connect to the Ollama host:port. Never raises."""
    try:
        with socket.create_connection((config.OLLAMA_HOST, config.OLLAMA_PORT), timeout):
            return True
    except OSError:
        return False


@app.get("/api/health", response_model=Health)
def health():
    """Liveness + backend info. Apple backend is on-device, so skip the probe
    and report reachable. Never raises."""
    apple = config.is_apple_model(config.CHAT_MODEL)
    reachable = True if apple else _ollama_reachable()
    return Health(
        ok=True,
        backend="apple" if apple else "ollama",
        model=config.CHAT_MODEL,
        reachable=reachable,
    )


# /api routers are registered BEFORE the catch-all static mount so /api wins.
app.include_router(chat_router)
app.include_router(conversations_router)

# In production the built frontend is served from frontend/dist; in dev that
# directory doesn't exist yet, so guard the mount and boot without it.
_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_dist), html=True), name="frontend")
