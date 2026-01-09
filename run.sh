#!/bin/bash
# AI Manipulation Detection & Mitigation - Hackathon Demo
# One command: ./run.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HATCAT_DIR="$SCRIPT_DIR/../HatCat"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "=========================================="
echo "AI Manipulation Detection & Mitigation"
echo "Hackathon Demo"
echo "=========================================="
echo ""

# Step 1: Check/clone HatCat
if [ ! -d "$HATCAT_DIR" ]; then
    echo "[1/5] Cloning HatCat repository..."
    git clone https://github.com/p0ss/HatCat.git "$HATCAT_DIR"
else
    echo "[1/5] HatCat already present at $HATCAT_DIR"
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
echo "Dashboard URL: http://localhost:8080"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Open browser (platform-specific, non-blocking)
if command -v xdg-open &> /dev/null; then
    (sleep 2 && xdg-open "http://localhost:8080") &
elif command -v open &> /dev/null; then
    (sleep 2 && open "http://localhost:8080") &
fi

# Start server
cd "$SCRIPT_DIR"
export PYTHONPATH="$HATCAT_DIR:$SCRIPT_DIR:$PYTHONPATH"
python -m uvicorn app.server.app:app --host 0.0.0.0 --port 8080
