#!/usr/bin/env bash
# Start FramePack Studio with low-VRAM optimizations (RTX 2060 / 6GB)
# Usage: ./start-framepack.sh [--inbrowser]

cd "$(dirname "$0")"

if [ ! -f "./venv/bin/activate" ]; then
  echo "Error: Virtual environment not found. Run setup first."
  exit 1
fi

source venv/bin/activate

echo "🚀 Starting FramePack Studio..."
echo "   VRAM mode: Low-VRAM (auto-detected)"
echo "   Gradio UI: http://localhost:7860"
echo ""

python studio.py "$@"
