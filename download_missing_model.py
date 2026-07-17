#!/usr/bin/env python3
"""
Download the missing FramePackI2V_HY model file (diffusion_pytorch_model-00003-of-00003.safetensors)
with retry logic, progress bar, and resume support.
"""
import os
import sys
import time

# Activate venv path
venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
if os.path.exists(venv_path):
    site_packages = [d for d in os.listdir(os.path.join(venv_path, "lib")) if d.startswith("python")][0]
    sys.path.insert(0, os.path.join(venv_path, "lib", site_packages, "site-packages"))

from huggingface_hub import hf_hub_download, HfApi, get_token
from huggingface_hub.utils import LocalTokenNotFoundError

REPO_ID = "lllyasviel/FramePackI2V_HY"
MISSING_FILE = "diffusion_pytorch_model-00003-of-00003.safetensors"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_download", "hub")

def get_file_size(repo_id, filename):
    """Get the remote file size."""
    try:
        api = HfApi()
        model_info = api.model_info(repo_id, files_metadata=True)
        for sibling in model_info.siblings:
            if sibling.rfilename == filename:
                return sibling.size
    except Exception as e:
        print(f"  Could not get file size from API: {e}")
    return None

def download_with_retry():
    """Download the missing file with retry logic."""
    print(f"Model: {REPO_ID}")
    print(f"File:  {MISSING_FILE}")
    print(f"Cache: {CACHE_DIR}")
    print()

    # Try to get file size
    remote_size = get_file_size(REPO_ID, MISSING_FILE)
    if remote_size:
        print(f"Remote size: {remote_size / 1e9:.2f} GB")
    print()

    max_retries = 5
    retry_delays = [5, 10, 30, 60, 120]  # Increasing backoff

    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}/{max_retries}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=MISSING_FILE,
                cache_dir=CACHE_DIR,
                resume_download=True,
                force_download=False,
                local_files_only=False,
            )
            print(f"\n✅ Download complete: {downloaded_path}")

            # Verify the file is non-zero
            actual_size = os.path.getsize(downloaded_path)
            print(f"   File size: {actual_size / 1e9:.2f} GB")
            if actual_size == 0:
                print("   ⚠️ File is zero bytes! Retrying...")
                os.remove(downloaded_path)
                continue
            return True

        except Exception as e:
            error_str = str(e)
            print(f"   ❌ Error: {error_str[:200]}")

            # Check if it's a flaky connection issue
            if attempt < max_retries:
                delay = retry_delays[min(attempt - 1, len(retry_delays) - 1)]
                print(f"   Retrying in {delay}s...")
                print()
                time.sleep(delay)
            else:
                print()
                print("All retries exhausted.")
                print()
                print("Trying alternative: direct URL download with requests...")
                return download_direct_with_requests()

    return False


def download_direct_with_requests():
    """Fallback: use requests directly with streaming and longer timeouts."""
    import requests

    url = f"https://huggingface.co/{REPO_ID}/resolve/main/{MISSING_FILE}"
    dest_dir = os.path.join(CACHE_DIR, f"models--{REPO_ID.replace('/', '--')}", "blobs")
    os.makedirs(dest_dir, exist_ok=True)

    temp_path = os.path.join(dest_dir, MISSING_FILE + ".download")

    print(f"Direct URL: {url}")
    print(f"Downloading to: {temp_path}")
    print()

    # Get HuggingFace token for authenticated requests
    hf_token = get_token()
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        print("Using HF auth token for download.")
    else:
        print("No HF token found. Download may fail for gated/restricted models.")

    # Use a session with longer timeouts
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
    })
    if hf_token:
        session.headers.update({"Authorization": f"Bearer {hf_token}"})

    try:
        # First do a HEAD request to check
        head_resp = session.head(url, timeout=30, allow_redirects=True)
        print(f"HTTP Status: {head_resp.status_code}")
        if head_resp.status_code != 200:
            print(f"  Headers: {dict(head_resp.headers)}")
            return False

        total_size = int(head_resp.headers.get("content-length", 0))
        print(f"Remote size: {total_size / 1e9:.2f} GB")

        # Check if we have a partial download
        downloaded_size = 0
        mode = "wb"
        if os.path.exists(temp_path):
            downloaded_size = os.path.getsize(temp_path)
            if downloaded_size > 0 and downloaded_size < total_size:
                print(f"Resuming from {downloaded_size / 1e9:.2f} GB")
                mode = "ab"
                session.headers.update({"Range": f"bytes={downloaded_size}-"})
            elif downloaded_size >= total_size:
                print("Already fully downloaded!")
                return True

        # Stream download with much longer timeout
        resp = session.get(url, stream=True, timeout=(30, 600), allow_redirects=True)

        if resp.status_code not in (200, 206):
            print(f"Bad status: {resp.status_code}")
            return False

        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        downloaded_so_far = downloaded_size

        with open(temp_path, mode) as f:
            start_time = time.time()
            last_log = 0

            for chunk in resp.iter_content(chunk_size=10 * 1024 * 1024):  # 10 MB chunks
                if chunk:
                    f.write(chunk)
                    downloaded_so_far += len(chunk)

                    # Log progress every 10 seconds
                    elapsed = time.time() - start_time
                    if elapsed - last_log >= 10:
                        pct = (downloaded_so_far / total_size * 100) if total_size else 0
                        speed = downloaded_so_far / elapsed / 1e6
                        print(f"  Progress: {downloaded_so_far / 1e9:.2f} GB / {total_size / 1e9:.2f} GB "
                              f"({pct:.1f}%) @ {speed:.1f} MB/s")
                        last_log = elapsed

        elapsed = time.time() - start_time
        final_size = os.path.getsize(temp_path)
        speed = final_size / elapsed / 1e6 if elapsed > 0 else 0
        print(f"\n✅ Download complete! ({final_size / 1e9:.2f} GB in {elapsed:.0f}s @ {speed:.1f} MB/s)")

        # Now compute the hash and rename
        print("Computing SHA256 hash...")
        import hashlib
        sha256 = hashlib.sha256()
        with open(temp_path, "rb") as f:
            while chunk := f.read(64 * 1024 * 1024):  # 64 MB chunks
                sha256.update(chunk)
        file_hash = sha256.hexdigest()

        # Rename to the blob hash
        final_path = os.path.join(dest_dir, file_hash)
        os.rename(temp_path, final_path)
        print(f"   Stored as blob: {file_hash}")
        print(f"   Final path: {final_path}")

        # Update the snapshot symlink
        snapshots_dir = os.path.join(CACHE_DIR, f"models--{REPO_ID.replace('/', '--')}", "snapshots")
        if os.path.exists(snapshots_dir):
            for snap in os.listdir(snapshots_dir):
                link_path = os.path.join(snapshots_dir, snap, MISSING_FILE)
                if os.path.islink(link_path):
                    os.unlink(link_path)
                    os.symlink(f"../../../blobs/{file_hash}", link_path)
                    print(f"   Updated symlink: {link_path} -> ../../../blobs/{file_hash}")

        return True

    except Exception as e:
        print(f"   ❌ Direct download failed: {e}")
        return False


def verify_cache():
    """Check the current cache state."""
    print("\n=== Current cache state ===")
    model_dir = os.path.join(CACHE_DIR, f"models--{REPO_ID.replace('/', '--')}")
    if os.path.exists(model_dir):
        blobs_dir = os.path.join(model_dir, "blobs")
        if os.path.exists(blobs_dir):
            total = 0
            for f in os.listdir(blobs_dir):
                fp = os.path.join(blobs_dir, f)
                if os.path.isfile(fp) and not f.endswith(".incomplete"):
                    size = os.path.getsize(fp)
                    if size > 1e6:
                        total += size
                        print(f"  {f[:16]}... {size / 1e9:.2f} GB")
            print(f"  Total: {total / 1e9:.2f} GB")
    print()


if __name__ == "__main__":
    verify_cache()
    success = download_with_retry()
    verify_cache()

    if success:
        print("\n✅ FramePackI2V_HY model download complete!")
        print("You can now restart the studio: source venv/bin/activate && python studio.py")
    else:
        print("\n❌ Download failed after all retries.")
        print("Try:")
        print("  1. Check your internet connection")
        print("  2. Try a different network/VPN")
        print("  3. Download manually from: https://huggingface.co/lllyasviel/FramePackI2V_HY")
        sys.exit(1)
