"""
Setup Routes - Model and lens pack download endpoints.
"""

import asyncio
import json
import sys
import torch
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# Shared path helpers
from app.utils.paths import (
    PROJECT_ROOT,
    ensure_hatcat_on_sys_path,
    load_project_config,
    resolve_hatcat_root,
)

CONFIG = load_project_config()
HATCAT_ROOT = resolve_hatcat_root(CONFIG)


def _add_hatcat_to_path():
    """Add HatCat src directory to sys.path for immediate use."""
    ensure_hatcat_on_sys_path(CONFIG)
    return bool(HATCAT_ROOT and HATCAT_ROOT.exists())

def check_hatcat_installed():
    """Check if HatCat is installed as a package."""
    try:
        import src.map
        return True
    except ImportError:
        # Try adding to path and check again
        if _add_hatcat_to_path():
            try:
                import src.map
                return True
            except ImportError:
                pass
        return False

def install_hatcat():
    """Try to install HatCat as an editable package."""
    import subprocess
    import importlib

    if not HATCAT_ROOT:
        return False, "HatCat path not configured. Set HATCAT_ROOT env var or hatcat_path in config.yaml."

    if not HATCAT_ROOT.exists():
        return False, f"HatCat not found at {HATCAT_ROOT}"

    try:
        # Run pip install -e
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", str(HATCAT_ROOT)],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            # If pip fails, try just adding to path directly
            _add_hatcat_to_path()

        # Force Python to recognize the new package
        # Clear any cached failed imports
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith('src.'):
                del sys.modules[mod_name]

        # Add to path regardless (belt and suspenders)
        _add_hatcat_to_path()

        # Verify it works
        try:
            import src.map
            return True, "HatCat installed and loaded successfully"
        except ImportError as e:
            return False, f"Installed but import still fails: {e}"

    except subprocess.TimeoutExpired:
        return False, "pip install timed out (300s)"
    except Exception as e:
        return False, f"Install failed: {str(e)}"

def ensure_hatcat():
    """Ensure HatCat is available for import."""
    if check_hatcat_installed():
        return True, "HatCat already installed"

    # Try to install it
    return install_hatcat()

router = APIRouter()


def get_state():
    from app.server.app import get_state
    return get_state()


@router.get("/status")
async def get_status():
    """Check setup status: model, lens pack, GPU availability."""
    state = get_state()

    # Check GPU
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    vram_gb = 0
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    # Check HatCat installation
    hatcat_linked = check_hatcat_installed()
    hatcat_path_exists = bool(HATCAT_ROOT and HATCAT_ROOT.exists())

    return {
        "model_ready": state.model_loaded,
        "lens_ready": state.lens_loaded,
        "hatcat_linked": hatcat_linked,
        "hatcat_path_exists": hatcat_path_exists,
        "hatcat_path": str(HATCAT_ROOT) if HATCAT_ROOT else None,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
    }


@router.post("/install-hatcat")
async def install_hatcat_endpoint():
    """Manually trigger HatCat installation."""
    success, message = install_hatcat()
    if success:
        return {"success": True, "message": message}
    else:
        raise HTTPException(status_code=500, detail=message)


async def model_download_generator() -> AsyncGenerator[str, None]:
    """Stream model download progress via SSE."""
    state = get_state()

    try:
        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting model download...'})}\n\n"
        await asyncio.sleep(0.1)

        # Import transformers
        yield f"data: {json.dumps({'type': 'progress', 'percent': 5, 'message': 'Loading transformers library...'})}\n\n"
        await asyncio.sleep(0.1)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = state.config.get("model", {}).get("name", "google/gemma-3-4b-pt")

        yield f"data: {json.dumps({'type': 'progress', 'percent': 10, 'message': f'Downloading tokenizer for {model_name}...'})}\n\n"
        await asyncio.sleep(0.1)

        # Load tokenizer
        state.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if state.tokenizer.pad_token is None:
            state.tokenizer.pad_token = state.tokenizer.eos_token

        yield f"data: {json.dumps({'type': 'progress', 'percent': 20, 'message': 'Tokenizer loaded!'})}\n\n"
        await asyncio.sleep(0.1)

        # Load model
        dtype = state.config.get("model", {}).get("dtype", "bfloat16")
        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

        yield f"data: {json.dumps({'type': 'progress', 'percent': 25, 'message': f'Downloading model weights (this may take a while)...'})}\n\n"
        await asyncio.sleep(0.1)

        yield f"data: {json.dumps({'type': 'status', 'message': f'Model: {model_name}, dtype: {dtype}'})}\n\n"
        await asyncio.sleep(0.1)

        state.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        yield f"data: {json.dumps({'type': 'progress', 'percent': 90, 'message': 'Model downloaded, setting to eval mode...'})}\n\n"
        await asyncio.sleep(0.1)

        state.model.eval()
        state.model_loaded = True

        # Report memory usage if GPU
        if torch.cuda.is_available():
            mem_used = torch.cuda.memory_allocated() / 1e9
            yield f"data: {json.dumps({'type': 'status', 'message': f'GPU memory used: {mem_used:.1f} GB'})}\n\n"

        yield f"data: {json.dumps({'type': 'progress', 'percent': 100, 'message': 'Model loaded successfully!'})}\n\n"
        yield f"data: {json.dumps({'type': 'complete', 'success': True})}\n\n"

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': f'Traceback: {tb[:500]}'})}\n\n"


@router.post("/download-model")
async def download_model():
    """Download and load the model with SSE progress streaming."""
    return StreamingResponse(
        model_download_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


async def lens_download_generator() -> AsyncGenerator[str, None]:
    """Stream lens pack download progress via SSE."""
    state = get_state()

    try:
        yield f"data: {json.dumps({'type': 'status', 'message': 'Starting lens pack download...'})}\n\n"
        await asyncio.sleep(0.1)  # Allow SSE to flush

        # Check if HatCat is installed
        yield f"data: {json.dumps({'type': 'progress', 'percent': 5, 'message': 'Checking HatCat installation...'})}\n\n"
        await asyncio.sleep(0.1)

        success, message = ensure_hatcat()
        if not success:
            yield f"data: {json.dumps({'type': 'error', 'message': message})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'progress', 'percent': 10, 'message': message})}\n\n"
        await asyncio.sleep(0.1)

        # Import HatCat components
        try:
            from src.map.registry import PackRegistry
            yield f"data: {json.dumps({'type': 'progress', 'percent': 20, 'message': 'Registry module loaded'})}\n\n"
        except ImportError as ie:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Failed to import HatCat registry: {ie}'})}\n\n"
            recommended_path = str(HATCAT_ROOT or '<your-hatcat-path>')
            yield f"data: {json.dumps({'type': 'status', 'message': f'Try running: pip install -e {recommended_path}'})}\n\n"
            return

        await asyncio.sleep(0.1)

        # Get config - check for local lens pack first
        lens_config = state.config.get("lens_pack", {})

        # Initialize paths
        lens_pack_path = None
        hierarchy_path = None

        # Use existing local lens pack if available (much faster than downloading)
        # The actual lens pack with .pt files is in src/lens_packs/
        local_pack_path = None
        local_hierarchy_path = None
        if HATCAT_ROOT and HATCAT_ROOT.exists():
            local_pack_path = HATCAT_ROOT / "src" / "lens_packs" / "gemma-3-4b-first-light-v1"
            local_hierarchy_path = HATCAT_ROOT / "concept_packs" / "first-light" / "hierarchy"

        if local_pack_path and local_pack_path.exists():
            yield f"data: {json.dumps({'type': 'progress', 'percent': 50, 'message': f'Using local lens pack: {local_pack_path}'})}\n\n"
            lens_pack_path = local_pack_path
            hierarchy_path = local_hierarchy_path if local_hierarchy_path and local_hierarchy_path.exists() else None
        else:
            # Fall back to download
            repo_id = lens_config.get("repo_id", "HatCatFTW/lens-gemma-3-4b-first-light-v1")
            pack_name = lens_config.get("name", "gemma-3-4b-first-light-v1")

            yield f"data: {json.dumps({'type': 'progress', 'percent': 30, 'message': f'Downloading {pack_name} from HuggingFace...'})}\n\n"
            await asyncio.sleep(0.1)

            # Pull lens pack with stdout capture
            registry = PackRegistry()
            yield f"data: {json.dumps({'type': 'progress', 'percent': 35, 'message': 'Contacting HuggingFace Hub...'})}\n\n"
            await asyncio.sleep(0.1)

            # Run download in thread with real-time output capture
            import threading
            import queue

            output_queue = queue.Queue()
            download_complete = threading.Event()
            download_error = [None]

            class QueueWriter:
                """Stdout wrapper that sends lines to a queue."""
                def __init__(self, q, original):
                    self.queue = q
                    self.original = original
                    self.buffer = ""

                def write(self, text):
                    self.original.write(text)  # Also write to terminal
                    self.buffer += text
                    while '\n' in self.buffer:
                        line, self.buffer = self.buffer.split('\n', 1)
                        if line.strip():
                            self.queue.put(line.strip())

                def flush(self):
                    self.original.flush()
                    if self.buffer.strip():
                        self.queue.put(self.buffer.strip())
                        self.buffer = ""

            def do_download():
                try:
                    old_stdout = sys.stdout
                    sys.stdout = QueueWriter(output_queue, old_stdout)
                    try:
                        registry.pull_lens_pack(pack_name, repo_id=repo_id)
                    finally:
                        sys.stdout.flush()
                        sys.stdout = old_stdout
                except Exception as e:
                    download_error[0] = e
                finally:
                    download_complete.set()

            thread = threading.Thread(target=do_download)
            thread.start()

            # Stream progress while downloading
            progress = 40
            while not download_complete.is_set():
                await asyncio.sleep(0.2)
                # Drain the queue
                while True:
                    try:
                        line = output_queue.get_nowait()
                        if line:
                            progress = min(progress + 1, 78)
                            yield f"data: {json.dumps({'type': 'progress', 'percent': progress, 'message': line[:120]})}\n\n"
                    except queue.Empty:
                        break

            thread.join()

            # Drain any remaining messages
            while True:
                try:
                    line = output_queue.get_nowait()
                    if line:
                        yield f"data: {json.dumps({'type': 'progress', 'percent': 79, 'message': line[:120]})}\n\n"
                except queue.Empty:
                    break

            if download_error[0]:
                raise download_error[0]

            lens_pack_path = registry.lens_packs_dir / pack_name

        yield f"data: {json.dumps({'type': 'progress', 'percent': 80, 'message': 'Lens pack ready!'})}\n\n"
        await asyncio.sleep(0.1)

        # Initialize lens manager if model is loaded
        if state.model_loaded and state.model is not None:
            yield f"data: {json.dumps({'type': 'progress', 'percent': 90, 'message': 'Initializing lens manager...'})}\n\n"
            await asyncio.sleep(0.1)

            from src.hat.monitoring.lens_manager import DynamicLensManager

            # lens_pack_path was set above (either from local or download)
            yield f"data: {json.dumps({'type': 'status', 'message': f'Lens pack at: {lens_pack_path}'})}\n\n"

            # Build kwargs for DynamicLensManager
            # Let the deployment manifest control base_layers and dynamic loading
            lens_manager_kwargs = {
                "lenses_dir": lens_pack_path,
                "load_threshold": 0.3,
                "max_loaded_lenses": 500,  # Match manifest's max_loaded_concepts
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "use_activation_lenses": True,
            }

            # Add hierarchy path if we have local concept metadata
            if hierarchy_path is not None and hierarchy_path.exists():
                lens_manager_kwargs["layers_data_dir"] = hierarchy_path
                yield f"data: {json.dumps({'type': 'status', 'message': f'Concept hierarchy at: {hierarchy_path}'})}\n\n"

            # Check for deployment manifest - this controls base_layers
            manifest_path = lens_pack_path / "deployment_manifest.json"
            if manifest_path.exists():
                lens_manager_kwargs["manifest_path"] = manifest_path
                yield f"data: {json.dumps({'type': 'status', 'message': 'Using deployment manifest for layer config'})}\n\n"

            state.lens_manager = DynamicLensManager(**lens_manager_kwargs)

            # DEBUG: Check what's loaded
            loaded_count = len(state.lens_manager.cache.loaded_lenses)
            print(f"[SETUP DEBUG] Loaded lenses: {loaded_count}")
            if loaded_count > 0:
                sample_keys = list(state.lens_manager.cache.loaded_lenses.keys())[:5]
                print(f"[SETUP DEBUG] Sample lens keys: {sample_keys}")

            # Check calibration was loaded
            if state.lens_manager.manifest and state.lens_manager.manifest.concept_calibration:
                cal_count = len(state.lens_manager.manifest.concept_calibration)
                yield f"data: {json.dumps({'type': 'status', 'message': f'Calibration loaded for {cal_count} concepts (scores normalized)'})}\n\n"
                print(f"[SETUP DEBUG] Calibration data loaded for {cal_count} concepts")
            else:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Warning: No calibration data - scores may be noisy'})}\n\n"
                print("[SETUP DEBUG] Warning: No calibration data loaded")

            # HushedGenerator sets lens_manager.last_detections directly, we just need
            # to ensure it's accessible. Initialize as empty list.
            state.lens_manager.last_detections = []

            # Save lens_pack_path to config for evaluation runner
            if "lens_pack" not in state.config:
                state.config["lens_pack"] = {}
            state.config["lens_pack"]["local_path"] = str(lens_pack_path)

            # Add concept_pack_path to config for auto-contrastive selection in HUSH
            # The concept pack is the parent directory of the hierarchy folder
            if hierarchy_path is not None and hierarchy_path.exists():
                concept_pack_path = hierarchy_path.parent
                state.config["concept_pack_path"] = str(concept_pack_path)
                yield f"data: {json.dumps({'type': 'status', 'message': f'Concept pack for contrastive steering: {concept_pack_path.name}'})}\n\n"

            loaded_count = len(state.lens_manager.cache.loaded_lenses)
            total_available = len(state.lens_manager.concept_metadata)
            yield f"data: {json.dumps({'type': 'status', 'message': f'Loaded {loaded_count} base lenses ({total_available} concepts available)'})}\n\n"

            state.lens_loaded = True

            yield f"data: {json.dumps({'type': 'progress', 'percent': 100, 'message': 'Lens pack ready!'})}\n\n"
        else:
            state.lens_loaded = True  # Mark as downloaded even if manager not initialized
            yield f"data: {json.dumps({'type': 'progress', 'percent': 100, 'message': 'Lens pack downloaded. Load model first to initialize lens manager.'})}\n\n"

        yield f"data: {json.dumps({'type': 'complete', 'success': True})}\n\n"

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        yield f"data: {json.dumps({'type': 'error', 'message': f'{str(e)}'})}\n\n"
        yield f"data: {json.dumps({'type': 'status', 'message': f'Traceback: {tb[:500]}'})}\n\n"


@router.post("/download-lens")
async def download_lens():
    """Download lens pack with SSE progress streaming."""
    return StreamingResponse(
        lens_download_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/initialize-generator")
async def initialize_generator():
    """Initialize the HushedGenerator after model and lens are loaded."""
    state = get_state()

    if not state.model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded")

    if not state.lens_loaded:
        raise HTTPException(status_code=400, detail="Lens pack not loaded")

    try:
        from src.hush.hush_integration import create_hushed_generator
        from src.hush.hush_controller import SafetyHarnessProfile, SimplexConstraint

        # Create USH profile for manipulation defense (Condition C)
        hush_config = state.config.get("hush", {}).get("manipulation_defense", {})
        constraints = []

        for c in hush_config.get("constraints", []):
            constraints.append(SimplexConstraint(
                simplex_term=c["simplex_term"],
                max_deviation=c.get("max_deviation", 0.5),
                target_pole=c.get("target_pole"),
                steering_strength=c.get("steering_strength", 0.5),
            ))

        ush_profile = SafetyHarnessProfile(
            profile_id="manipulation-defense",
            profile_type="ush",
            issuer_tribe_id="hackathon-demo",
            version="1.0",
            constraints=constraints,
        ) if constraints else None

        state.generator, _ = create_hushed_generator(
            model=state.model,
            tokenizer=state.tokenizer,
            lens_manager=state.lens_manager,
            ush_profile=ush_profile,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        return {"success": True, "message": "Generator initialized"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
