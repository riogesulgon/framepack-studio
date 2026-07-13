#!/usr/bin/env bash
# Stop FramePack Studio
# Usage: ./stop-framepack.sh

PID=$(pgrep -f "python studio.py" 2>/dev/null | head -1)

if [ -z "$PID" ]; then
  echo "FramePack Studio is not running."
  exit 0
fi

echo "Stopping FramePack Studio (PID: $PID)..."
kill "$PID"
sleep 1

# Force kill if still running
if kill -0 "$PID" 2>/dev/null; then
  echo "Force stopping..."
  kill -9 "$PID" 2>/dev/null
fi

echo "✅ Stopped."
