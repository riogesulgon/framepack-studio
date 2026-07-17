#!/usr/bin/env python3
"""
Clean orphaned blob files from the HuggingFace model cache.

Orphaned blobs are files in hf_download/hub/models--*/blobs/ that are NOT
referenced by any symlink in the corresponding snapshots/ directories.
These accumulate over time as you switch between model versions/revisions.

Usage:
    # Dry-run (default) — only reports what would be deleted
    python scripts/clean_hf_blob_cache.py

    # Delete orphans (pass --delete or set DRY_RUN=false)
    python scripts/clean_hf_blob_cache.py --delete

    # Point to a specific HF_HOME
    python scripts/clean_hf_blob_cache.py --hf-home /custom/path

    # Verbose — list each orphaned file
    python scripts/clean_hf_blob_cache.py --delete --verbose

    # Include .incomplete files (active/stale downloads)
    python scripts/clean_hf_blob_cache.py --delete --include-incomplete
"""

import argparse
import os
import sys
from pathlib import Path


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes >= 1073741824:
        return f"{size_bytes / 1073741824:.1f} GB"
    elif size_bytes >= 1048576:
        return f"{size_bytes / 1048576:.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def find_hf_home() -> str:
    """
    Resolve HF_HOME by checking, in order:
    1. HF_HOME environment variable
    2. Default project path relative to this script
    3. Default HF cache in user home
    """
    # 1. Check env var
    env_hf = os.environ.get("HF_HOME")
    if env_hf and os.path.isdir(env_hf):
        return env_hf

    # 2. Check project-relative path (this script lives in scripts/)
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    project_hf = project_dir / "hf_download"
    if project_hf.is_dir():
        return str(project_hf)

    # 3. Check default HF cache
    home_hf = Path.home() / ".cache" / "huggingface"
    if home_hf.is_dir():
        return str(home_hf)

    return str(project_hf)  # fallback, will report not found


def scan_orphans(hub_path: str, include_incomplete: bool = False) -> dict:
    """
    Scan all models in hub_path and identify orphaned blob files.

    Returns a dict with:
        model_dirs: list of model directory names scanned
        total_orphans: count
        total_orphan_size: bytes
        orphans: list of dicts with path, size, model
    """
    result = {
        "model_dirs": [],
        "total_orphans": 0,
        "total_orphan_size": 0,
        "orphans": [],
    }

    model_dirs = [
        d for d in os.listdir(hub_path)
        if d.startswith("models--") and os.path.isdir(os.path.join(hub_path, d))
    ]
    result["model_dirs"] = model_dirs

    for model_dir in model_dirs:
        model_path = os.path.join(hub_path, model_dir)
        blobs_dir = os.path.join(model_path, "blobs")
        snapshots_dir = os.path.join(model_path, "snapshots")

        if not os.path.isdir(blobs_dir):
            continue

        # Collect referenced blob hashes from ALL snapshot symlinks
        referenced_hashes: set[str] = set()
        if os.path.isdir(snapshots_dir):
            try:
                for snapshot in os.listdir(snapshots_dir):
                    snapshot_path = os.path.join(snapshots_dir, snapshot)
                    if not os.path.isdir(snapshot_path):
                        continue
                    for entry in os.listdir(snapshot_path):
                        entry_path = os.path.join(snapshot_path, entry)
                        if os.path.islink(entry_path):
                            target = os.readlink(entry_path)
                            blob_hash = os.path.basename(target)
                            if blob_hash:
                                referenced_hashes.add(blob_hash)
            except OSError as e:
                print(f"  ⚠️  Error reading snapshots in {model_dir}: {e}", file=sys.stderr)
                continue

        # Check each blob
        try:
            for blob_file in os.listdir(blobs_dir):
                blob_path = os.path.join(blobs_dir, blob_file)

                # Skip directories
                if not os.path.isfile(blob_path):
                    continue

                # Handle .incomplete files
                if blob_file.endswith(".incomplete"):
                    if not include_incomplete:
                        continue

                if blob_file not in referenced_hashes:
                    file_size = os.path.getsize(blob_path)
                    result["total_orphans"] += 1
                    result["total_orphan_size"] += file_size
                    result["orphans"].append(
                        {
                            "path": blob_path,
                            "size": file_size,
                            "model": model_dir,
                            "is_incomplete": blob_file.endswith(".incomplete"),
                        }
                    )
        except OSError as e:
            print(f"  ⚠️  Error reading blobs in {model_dir}: {e}", file=sys.stderr)
            continue

    return result


def delete_orphans(orphans: list[dict], verbose: bool = False) -> tuple[int, int]:
    """Delete orphaned blobs. Returns (deleted_count, deleted_bytes)."""
    deleted_count = 0
    deleted_bytes = 0
    failed_count = 0

    for orphan in orphans:
        try:
            os.remove(orphan["path"])
            deleted_count += 1
            deleted_bytes += orphan["size"]
            if verbose:
                tag = " [incomplete]" if orphan["is_incomplete"] else ""
                print(f"    🗑️  Deleted: {orphan['path']}{tag}")
        except OSError as e:
            failed_count += 1
            print(f"    ❌ Failed to delete {orphan['path']}: {e}", file=sys.stderr)

    if failed_count:
        print(f"\n  ⚠️  {failed_count} file(s) could not be deleted (permissions?)")

    return deleted_count, deleted_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Clean orphaned blob files from HuggingFace model cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hf-home",
        default=None,
        help="Path to HF_HOME (default: auto-detect from env, project dir, or ~/.cache/huggingface)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        default=False,
        help="Actually delete orphaned blobs (default is dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Explicit dry-run mode (overrides --delete if both passed)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="List each orphaned file individually",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        default=False,
        help="Include .incomplete files in orphan detection (active downloads are skipped by default)",
    )

    args = parser.parse_args()

    # Resolve HF_HOME
    hf_home = args.hf_home or find_hf_home()
    hub_path = os.path.join(hf_home, "hub")

    print(f"🔍 HF_HOME: {hf_home}")
    print(f"🔍 Hub dir: {hub_path}")

    if not os.path.isdir(hub_path):
        print(f"❌ Hub directory not found: {hub_path}", file=sys.stderr)
        sys.exit(1)

    # Scan for orphans
    print("\n📂 Scanning model cache for orphaned blobs...")
    result = scan_orphans(hub_path, include_incomplete=args.include_incomplete)

    if not result["model_dirs"]:
        print("  No model directories found (nothing to scan).")
        sys.exit(0)

    print(f"  Models scanned: {len(result['model_dirs'])}")
    print(f"  Orphaned blobs: {result['total_orphans']}")

    if result["total_orphans"] == 0:
        print("\n✅ No orphaned blobs found. Cache is clean!")
        sys.exit(0)

    total_size_str = format_size(result["total_orphan_size"])
    print(f"  Total wasted:   {total_size_str}")

    # List orphans if verbose
    if args.verbose:
        print("\n📋 Orphaned files:")
        for orphan in sorted(result["orphans"], key=lambda o: -o["size"]):
            tag = " [incomplete]" if orphan["is_incomplete"] else ""
            print(f"    {format_size(orphan['size']):>10}  {orphan['path']}{tag}")

    # Delete or dry-run
    is_dry_run = args.dry_run or not args.delete
    print()

    if is_dry_run:
        print(f"🚫 Dry-run mode — no files deleted.")
        print(f"   Run with --delete to remove {result['total_orphans']} file(s) ({total_size_str}).")
    else:
        print("🗑️  Deleting orphaned blobs...")
        deleted_count, deleted_bytes = delete_orphans(
            result["orphans"], verbose=args.verbose
        )
        freed_str = format_size(deleted_bytes)
        print(f"\n✅ {deleted_count} file(s) deleted ({freed_str} freed)")

    # Per-model breakdown
    if args.verbose:
        print("\n📊 Per-model breakdown:")
        model_totals: dict[str, dict] = {}
        for orphan in result["orphans"]:
            m = orphan["model"]
            if m not in model_totals:
                model_totals[m] = {"count": 0, "size": 0}
            model_totals[m]["count"] += 1
            model_totals[m]["size"] += orphan["size"]
        for model, stats in sorted(model_totals.items(), key=lambda x: -x[1]["size"]):
            pretty_name = model.replace("models--", "").replace("--", "/")
            print(f"    {pretty_name:50s}  {stats['count']:4d} files  {format_size(stats['size'])}")


if __name__ == "__main__":
    main()
