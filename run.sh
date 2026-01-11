#!/bin/bash
# AI Manipulation Detection & Mitigation - Hackathon Demo
# One command: ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Allow overriding the HatCat checkout location via env vars
DEFAULT_HATCAT_DIR="$SCRIPT_DIR/../HatCat"
HATCAT_DIR="${HATCAT_DIR:-${HATCAT_ROOT:-$DEFAULT_HATCAT_DIR}}"
HATCAT_BRANCH="${HATCAT_BRANCH:-main}"
export HATCAT_ROOT="$HATCAT_DIR"

# Allow overriding host/port from env or config.yaml (if PyYAML available)
CONFIG_SERVER_HOST=""
CONFIG_SERVER_PORT=""
if command -v python3 >/dev/null 2>&1; then
    CONFIG_VALUES=$(python3 - <<'PY'
cfg_host = ""
cfg_port = ""
try:
    import pathlib
    import yaml  # type: ignore

    config_path = pathlib.Path(__file__).resolve().parent / "config.yaml"
    if config_path.exists():
        data = yaml.safe_load(config_path.read_text()) or {}
        server_cfg = data.get("server", {}) or {}
        cfg_host = str(server_cfg.get("host", ""))
        cfg_port = str(server_cfg.get("port", ""))
except Exception:
    pass

print(cfg_host)
print(cfg_port)
PY
)
    IFS=$'\n' read -r CONFIG_SERVER_HOST CONFIG_SERVER_PORT <<EOF
${CONFIG_VALUES}
EOF
fi

DEFAULT_SERVER_HOST=${DEFAULT_SERVER_HOST:-0.0.0.0}
DEFAULT_SERVER_PORT=${DEFAULT_SERVER_PORT:-8080}
SERVER_HOST="${SERVER_HOST:-${CONFIG_SERVER_HOST:-$DEFAULT_SERVER_HOST}}"
SERVER_PORT="${SERVER_PORT:-${CONFIG_SERVER_PORT:-$DEFAULT_SERVER_PORT}}"
PUBLIC_DASHBOARD_URL="${PUBLIC_DASHBOARD_URL:-}"
PUBLIC_DASHBOARD_HOST="${PUBLIC_DASHBOARD_HOST:-}"
PUBLIC_DASHBOARD_PORT="${PUBLIC_DASHBOARD_PORT:-}" # optional override when forwarded port differs
PUBLIC_DASHBOARD_SCHEME="${PUBLIC_DASHBOARD_SCHEME:-http}"

DASHBOARD_HOST_DISPLAY="$SERVER_HOST"
if [ "$DASHBOARD_HOST_DISPLAY" = "0.0.0.0" ] || [ "$DASHBOARD_HOST_DISPLAY" = "::" ]; then
    DASHBOARD_HOST_DISPLAY="127.0.0.1"
fi
DASHBOARD_PORT_DISPLAY="$SERVER_PORT"
DASHBOARD_URL="http://${DASHBOARD_HOST_DISPLAY}:${DASHBOARD_PORT_DISPLAY}"

if [ -n "$PUBLIC_DASHBOARD_URL" ]; then
    DASHBOARD_URL="$PUBLIC_DASHBOARD_URL"
elif [ -n "$PUBLIC_DASHBOARD_HOST" ]; then
    port_segment="${PUBLIC_DASHBOARD_PORT:-$SERVER_PORT}"
    DASHBOARD_URL="${PUBLIC_DASHBOARD_SCHEME}://${PUBLIC_DASHBOARD_HOST}:${port_segment}"
fi
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=========================================="
echo "AI Manipulation Detection & Mitigation"
echo "Hackathon Demo"
echo "=========================================="
echo ""

# Step 1: Check/clone HatCat
if [ ! -d "$HATCAT_DIR" ]; then
    echo "[1/5] Cloning HatCat repository (branch $HATCAT_BRANCH)..."
    git clone --branch "$HATCAT_BRANCH" https://github.com/p0ss/HatCat.git "$HATCAT_DIR"
elif [ ! -d "$HATCAT_DIR/.git" ]; then
    echo "[1/5] HatCat directory exists but is not a git repo: $HATCAT_DIR"
    echo "      Remove or set HATCAT_DIR to a valid clone."
else
    echo "[1/5] HatCat already present at $HATCAT_DIR"
    if [ "${HATCAT_UPDATE:-0}" != "0" ]; then
        echo "      Updating HatCat (branch $HATCAT_BRANCH)..."
        git -C "$HATCAT_DIR" fetch --all --prune
        if git -C "$HATCAT_DIR" rev-parse --verify "$HATCAT_BRANCH" >/dev/null 2>&1; then
            git -C "$HATCAT_DIR" checkout "$HATCAT_BRANCH" >/dev/null 2>&1 || true
        fi
        if ! git -C "$HATCAT_DIR" pull --ff-only; then
            echo "      Fast-forward failed, resetting to origin/$HATCAT_BRANCH"
            git -C "$HATCAT_DIR" fetch origin "$HATCAT_BRANCH"
            git -C "$HATCAT_DIR" checkout "$HATCAT_BRANCH"
            git -C "$HATCAT_DIR" reset --hard "origin/$HATCAT_BRANCH"
        fi
    fi
fi

# Step 2: Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "[2/5] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[2/5] Virtual environment exists"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Step 3: Install dependencies
echo "[3/5] Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Install HatCat as editable if not already
if ! python -c "import src.hush" 2>/dev/null; then
    echo "      Installing HatCat..."
    pip install -q -e "$HATCAT_DIR"
fi

# Step 4: Check GPU
echo "[4/5] Checking system..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'      GPU: {gpu_name} ({vram:.1f} GB VRAM)')
else:
    print('      WARNING: No GPU detected. Running on CPU will be slow.')
"

# Step 5: Launch server
echo "[5/5] Launching dashboard..."
echo ""
echo "=========================================="
echo "Dashboard URL: $DASHBOARD_URL"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Open browser (platform-specific, non-blocking)
if command -v xdg-open &> /dev/null; then
    (sleep 2 && xdg-open "$DASHBOARD_URL") &
elif command -v open &> /dev/null; then
    (sleep 2 && open "$DASHBOARD_URL") &
fi

# Start server
cd "$SCRIPT_DIR"
export PYTHONPATH="$HATCAT_DIR:$SCRIPT_DIR:$PYTHONPATH"
python -m uvicorn app.server.app:app --host "$SERVER_HOST" --port "$SERVER_PORT"
