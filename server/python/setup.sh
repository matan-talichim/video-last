#!/bin/bash
# Setup script for presenter detection Python environment
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
MODELS_DIR="$SCRIPT_DIR/models"
MODEL_FILE="$MODELS_DIR/face_landmarker_v2_with_blendshapes.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

echo "=== Setting up Python environment ==="

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment already exists."
fi

# Activate and install dependencies
echo "Installing Python dependencies..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q

# Download FaceLandmarker model
mkdir -p "$MODELS_DIR"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading FaceLandmarker model..."
    curl -L -o "$MODEL_FILE" "$MODEL_URL"
    echo "Model downloaded to $MODEL_FILE"
else
    echo "FaceLandmarker model already exists."
fi

echo "=== Python environment ready ==="
