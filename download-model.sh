#!/usr/bin/bash
# Download lllyasviel/FramePackI2V_HY with resume support.
# Logs progress, shows cache state, disk space.
# Safe to Ctrl+C and re-run — resumes where it left off.

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG="$DIR/download-model.log"
VENV="$DIR/venv"
HUB_DIR="$DIR/hf_download/hub"
MODEL_DIR="$HUB_DIR/models--lllyasviel--FramePackI2V_HY"

log()    { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }
header() { echo "" | tee -a "$LOG"; echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG"; echo "$*" | tee -a "$LOG"; echo "═══════════════════════════════════════════════════════════════════" | tee -a "$LOG"; }

header "FramePackI2V_HY Model Downloader"

# ── venv check ──
if [ ! -d "$VENV" ]; then
    log "ERROR: venv not found at $VENV"
    exit 1
fi
source "$VENV/bin/activate"

# ── Disk space ──
header "System Status"
log "Disk space: $(df -h "$DIR" | awk 'NR==2{print $3" used / "$2" total ("$5") — "$4" free"}')"
log "GPU VRAM:   $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "%.1f/%.0f GB", $1/1024, $2/1024}')"
log "Internet:   $(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 5 https://huggingface.co 2>/dev/null) OK"

# ── Show what's already cached ──
header "Already Cached (blobs)"
if [ -d "$MODEL_DIR/blobs" ]; then
    ALL_GOOD=true
    TOTAL_GB=0
    for f in "$MODEL_DIR/blobs"/*; do
        name="$(basename "$f")"
        if [[ "$name" == *.incomplete ]]; then
            log "  ⚠  $name  (INCOMPLETE — will resume)"
            ALL_GOOD=false
        elif [ -f "$f" ]; then
            size_gb=$(stat --printf="%s" "$f" 2>/dev/null | awk '{printf "%.2f", $1/1073741824}')
            TOTAL_GB=$(echo "$TOTAL_GB + $size_gb" | bc 2>/dev/null || echo "$TOTAL_GB")
            log "  ✓  ${name:0:16}...  ${size_gb} GB"
        fi
    done
    log "  Total cached: $(printf "%.1f" "$TOTAL_GB") GB"
    # The full model is 9.3 + 9.4 + 5.8 = ~24.5 GB
    NEEDED=$(echo "24.5 - $TOTAL_GB" | bc -l 2>/dev/null || echo "~5.8")
    log "  Remaining:    ~${NEEDED} GB"
else
    log "  (nothing cached yet)"
fi

# ── Snapshot symlinks ──
header "Snapshot Files"
if [ -d "$MODEL_DIR/snapshots" ]; then
    for snap in "$MODEL_DIR/snapshots"/*/; do
        snapname="$(basename "$snap")"
        log "  Snapshot: ${snapname:0:20}..."
        for f in "$snap"*.safetensors; do
            if [ -L "$f" ]; then
                target="$(readlink "$f")"
                filesize=$(stat --printf="%s" "$f" 2>/dev/null | awk '{printf "%.2f", $1/1073741824}')
                log "    ✓  $(basename "$f")"
            elif [ -f "$f" ]; then
                log "    ?  $(basename "$f")  (not a symlink)"
            fi
        done
        for f in "$snap"*.json; do
            [ -f "$f" ] && log "    ✓  $(basename "$f")"
        done
    done
else
    log "  (no snapshots yet)"
fi

# ── Check what file 3/3 status is ──
header "File 3/3 Status"
# The index file tells us the filename and expected size
INDEX_FILE="$MODEL_DIR/snapshots"/*/"diffusion_pytorch_model.safetensors.index.json" 2>/dev/null
if ls $INDEX_FILE 2>/dev/null; then
    INDEX_FILE=$(ls $INDEX_FILE 2>/dev/null | head -1)
    if [ -f "$INDEX_FILE" ]; then
        log "  Index file found: $(basename "$(dirname "$(dirname "$INDEX_FILE")")")"
        # Check if there's already a symlink for file 3
        SNAP_DIR="$(dirname "$INDEX_FILE")"
        FILE3_SYMLINK="$SNAP_DIR/transformer/diffusion_pytorch_model-00003-of-00003.safetensors" 2>/dev/null
        # Actually the files might be in the snapshot root, not transformer subfolder
        FILE3_SYMLINK="$SNAP_DIR/diffusion_pytorch_model-00003-of-00003.safetensors"
    fi
fi

if [ -L "$FILE3_SYMLINK" ] 2>/dev/null; then
    TARGET=$(readlink "$FILE3_SYMLINK")
    if [ -f "$TARGET" ]; then
        SIZE=$(stat --printf="%s" "$TARGET" 2>/dev/null | awk '{printf "%.2f GB", $1/1073741824}')
        log "  ✅ File 3/3 already complete! ($SIZE)"
        log "  No download needed."
        echo ""
        log "You can start the studio:  source venv/bin/activate && python studio.py"
        exit 0
    else
        log "  🔶 File 3/3 symlink exists but blob is missing (stale)"
    fi
else
    log "  ⬇  File 3/3 not downloaded yet (5.79 GB needed)"
fi

# ── Run the download ──
header "Downloading"
log "Starting hf download of lllyasviel/FramePackI2V_HY..."
log "(Progress bars and ETA will appear below. Ctrl+C safe — just re-run.)"
echo "" | tee -a "$LOG"

hf download lllyasviel/FramePackI2V_HY --cache-dir "$HUB_DIR" 2>&1 | tee -a "$LOG"
EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG"

# ── Result ──
header "Result"
if [ $EXIT_CODE -eq 0 ]; then
    log "✅ SUCCESS — FramePackI2V_HY fully downloaded!"
    echo "" | tee -a "$LOG"
    # Show final disk space
    log "Disk space: $(df -h "$DIR" | awk 'NR==2{print $3" used / "$2" total ("$5") — "$4" free"}')"
    echo "" | tee -a "$LOG"
    log "Ready. Start the studio:"
    log "  source venv/bin/activate && python studio.py"
else
    log "❌ FAILED (exit code $EXIT_CODE)"
    log "   This is usually a network timeout. Just re-run to resume:"
    log "   ./download-model.sh"
    log ""
    log "   If it keeps failing, try:"
    log "   1. Check your internet connection"
    log "   2. Use a VPN or different network"
    log "   3. Download manually from https://huggingface.co/lllyasviel/FramePackI2V_HY"
fi

exit $EXIT_CODE
