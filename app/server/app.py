"""
AI Manipulation Detection & Mitigation - FastAPI Server

Main application entry point for the hackathon demo dashboard.
"""

import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.utils.paths import (
    PROJECT_ROOT,
    ensure_hatcat_on_sys_path,
    load_project_config,
)

PROJECT_CONFIG = load_project_config()
HATCAT_ROOT = ensure_hatcat_on_sys_path(PROJECT_CONFIG)
sys.path.insert(0, str(PROJECT_ROOT))

from app.server.routes import setup, evaluation, results, compliance


# Run state for persistence across page refreshes
class RunState:
    def __init__(self):
        self.run_id = None
        self.status = "idle"  # idle, running, complete, aborted, error
        self.started_at = None
        self.completed_at = None
        self.total_episodes = 0
        self.completed_episodes = 0
        self.current_episode = None
        self.current_condition = None
        self.suite_id = None
        self.suite_name = None
        self.conditions = []
        self.episode_ids = []
        # Store completed results for display
        self.episode_results = []  # List of {episode_id, condition, result}
        self.tokens_buffer = []  # Recent tokens for current episode
        self.summary = None
        self.error_message = None


# Global state
class AppState:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.lens_manager = None
        self.generator = None
        self.model_loaded = False
        self.lens_loaded = False
        self.run_state = RunState()
        self.abort_requested = False


state = AppState(PROJECT_CONFIG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    print("Starting AI Manipulation Detection Server...")

    # Ensure output directories exist
    for dir_key in ["results_dir", "audit_logs_dir", "reports_dir"]:
        dir_path = PROJECT_ROOT / state.config.get("output", {}).get(dir_key, f"outputs/{dir_key}")
        dir_path.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="AI Manipulation Detection & Mitigation",
    description="Hackathon demo for detecting and mitigating AI manipulation behaviors",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = PROJECT_ROOT / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Include route modules
app.include_router(setup.router, prefix="/api/setup", tags=["Setup"])
app.include_router(evaluation.router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(results.router, prefix="/api/results", tags=["Results"])
app.include_router(compliance.router, prefix="/api/compliance", tags=["Compliance"])


@app.get("/")
async def root():
    """Serve the main dashboard."""
    index_path = PROJECT_ROOT / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "AI Manipulation Detection API", "docs": "/docs"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": state.model_loaded,
        "lens_loaded": state.lens_loaded,
    }


# Make state accessible to routes
def get_state() -> AppState:
    return state
