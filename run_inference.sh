#!/bin/bash
set -e

# Configuration
VENV_DIR="venv"
REQUIREMENTS="requirements.txt"
SCRIPT="inference.py"

echo "==========================================="
echo "Solar Panel Detection - Automation Script"
echo "==========================================="

# 1. Check/Create Virtual Environment
if [ -d "$VENV_DIR" ]; then
    echo "[*] Venv found."
else
    echo "[*] Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# 2. Activate Venv
echo "[*] Activating virtual environment..."
source $VENV_DIR/bin/activate

# 3. Install Dependencies
if [ -f "$REQUIREMENTS" ]; then
    echo "[*] Installing dependencies..."
    pip install --upgrade pip
    pip install -r $REQUIREMENTS
else
    echo "[!] Error: $REQUIREMENTS not found!"
    exit 1
fi

# 4. Run Inference
echo "[*] Running inference..."
if [ -f "$SCRIPT" ]; then
    # Pass all arguments to the python script
    python $SCRIPT "$@"
else
    echo "[!] Error: $SCRIPT not found!"
    exit 1
fi

echo "==========================================="
echo "Inference Complete."
echo "==========================================="
