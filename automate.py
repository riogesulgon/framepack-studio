#!/usr/bin/env python3
"""
FramePack Studio — Automation Script

Programmatically submit video generation jobs to a running FramePack Studio
(Gradio) instance via its built-in API.

Usage:
    # Generate from an image with default parameters
    python automate.py --input-image ~/cat.png --prompt "[1s: A cat waves hello]"

    # Generate from a video with custom parameters
    python automate.py --input-video ~/input.mp4 --model-type Video \\
        --prompt "[1s: A cat walks] [3s: The cat jumps]" --steps 30 --seed 42

    # Full control with config file and overrides
    python automate.py --config my_preset.json --seed 9999 --verbose

    # Watch for results without blocking forever
    python automate.py --input-image ~/cat.png --timeout 600 --poll-interval 10

Config file format (JSON):
    {
        "model_type": "Original",
        "prompt": "[1s: A cat walks across the room]",
        "steps": 20,
        "cfg": 6.0,
        "width": 640,
        "height": 640,
        "seed": 2500
    }

CLI args always override config file values.
"""

import argparse
import json
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from modules.automate_config import (
    PARAMETER_META,
    MODEL_TYPES,
    LATENT_TYPES,
    CACHE_TYPES,
    build_params_list,
    match_endpoint_by_input_count,
    get_parameter_count,
)


# ──────────────────────────────────────────────
# FramePack Client
# ──────────────────────────────────────────────

class FramePackClient:
    """Client for interacting with a running FramePack Studio instance."""

    def __init__(self, server_url: str = "http://localhost:7860", verbose: bool = False):
        self.server = server_url.rstrip("/")
        self.verbose = verbose
        self._session = requests.Session()
        self._endpoint: Optional[str] = None
        self._output_dir: Optional[str] = None  # server-side output dir, discovered at runtime

    def log(self, msg: str):
        if self.verbose:
            print(f"[FPClient] {msg}", file=sys.stderr)

    # ── Server Health ──

    def check_server(self) -> bool:
        """Verify the server is reachable."""
        try:
            r = self._session.get(f"{self.server}/", timeout=10)
            r.raise_for_status()
            return True
        except requests.RequestException as e:
            self.log(f"Server unreachable: {e}")
            return False

    # ── Endpoint Discovery ──

    def discover_endpoint(self, lora_count: int = 0) -> str:
        """
        Discover the correct Gradio API endpoint for job submission.

        First checks /gradio_api/info, then falls back to /api/predict/.
        Uses input-parameter count matching to identify the correct endpoint.
        """
        # Expected parameter count: model_type + len(PARAMETER_META) + 1 (lora_weights_dict)
        target_count = get_parameter_count(lora_count)

        # Try /gradio_api/info (Gradio 5.x)
        info_url = f"{self.server}/gradio_api/info"
        self.log(f"Discovering endpoint via {info_url} (target {target_count} params)")

        try:
            r = self._session.get(info_url, timeout=10)
            if r.status_code == 200:
                info = r.json()
                endpoint = match_endpoint_by_input_count(info, target_count)
                if endpoint:
                    self._endpoint = endpoint
                    self.log(f"Discovered endpoint: {endpoint} ({target_count} params)")
                    return endpoint

                # Fallback: try permissive matching (±3 params to account for LoRA variance)
                for name, ep_info in info.get("named_endpoints", {}).items():
                    params = ep_info.get("parameters", [])
                    if abs(len(params) - target_count) <= 3:
                        self.log(f"Permissive match: {name} ({len(params)} params)")
                        self._endpoint = name
                        return name
                for idx, ep_info in info.get("unnamed_endpoints", {}).items():
                    params = ep_info.get("parameters", [])
                    if abs(len(params) - target_count) <= 3:
                        self.log(f"Permissive match (unnamed): /{idx} ({len(params)} params)")
                        self._endpoint = f"/{idx}"
                        return f"/{idx}"
        except Exception as e:
            self.log(f"Endpoint discovery via /gradio_api/info failed: {e}")

        # Fallback: try common Gradio endpoints
        for candidate in ["/api/predict/", "/api/predict", "/gradio_api/predict"]:
            try:
                r = self._session.get(f"{self.server}{candidate}", timeout=5)
                if r.status_code in (200, 405, 422):
                    self.log(f"Using fallback endpoint: {candidate}")
                    self._endpoint = candidate
                    return candidate
            except requests.RequestException:
                continue

        raise RuntimeError(
            f"Could not discover Gradio API endpoint. "
            f"Check that the server is running at {self.server} "
            f"and that a generation job has been submitted at least once through the UI."
        )

    # ── File Upload ──

    def upload_file(self, file_path: str) -> str:
        """
        Upload a file (image/video) to the Gradio server.

        Returns the server-side file path to pass in the API call.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(file_path)
        self.log(f"Uploading {file_path} ({os.path.getsize(file_path):,} bytes)")

        # Try /gradio_api/upload first (Gradio 5.x)
        for upload_url in [
            f"{self.server}/gradio_api/upload",
            f"{self.server}/upload",
        ]:
            try:
                with open(file_path, "rb") as f:
                    r = self._session.post(
                        upload_url,
                        files={"files": (filename, f, self._guess_mime(file_path))},
                        timeout=120,
                    )
                if r.status_code == 200:
                    data = r.json()
                    self.log(f"Upload response: {data}")
                    # Handle various response formats
                    if isinstance(data, list):
                        if isinstance(data[0], dict):
                            # [{"name": "...", "data": null, "is_file": true}]
                            return data[0].get("name", "")
                        elif isinstance(data[0], str):
                            # ["/tmp/gradio/.../file.png"]
                            return data[0]
                    elif isinstance(data, dict):
                        # {"name": "...", "data": null, "is_file": true}
                        return data.get("name", "")
                    elif isinstance(data, str):
                        return data
            except Exception as e:
                self.log(f"Upload to {upload_url} failed: {e}")
                continue

        raise RuntimeError(f"Failed to upload {file_path} to {self.server}")

    @staticmethod
    def _guess_mime(path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".avi": "video/x-msvideo",
        }.get(ext, "application/octet-stream")

    # ── Discover server output directory ──

    def discover_output_dir(self) -> str:
        """
        Determine the server's output directory by checking common locations.
        Returns the path to watch for result videos.
        """
        # Default output dirs in order of likelihood
        candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
            os.path.join(os.getcwd(), "outputs"),
        ]

        # Try to read from settings.json if accessible
        settings_paths = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".framepack", "settings.json"),
            os.path.join(os.getcwd(), ".framepack", "settings.json"),
        ]
        for sp in settings_paths:
            if os.path.isfile(sp):
                try:
                    with open(sp) as f:
                        settings = json.load(f)
                    if "output_dir" in settings:
                        od = settings["output_dir"]
                        if os.path.isdir(od):
                            candidates.insert(0, od)
                except (json.JSONDecodeError, OSError):
                    pass

        for c in candidates:
            if os.path.isdir(c):
                self._output_dir = c
                self.log(f"Using output directory: {c}")
                return c

        # Fallback: create it
        fallback = candidates[0]
        os.makedirs(fallback, exist_ok=True)
        self._output_dir = fallback
        return fallback

    # ── Job Submission ──

    def submit_job(self, params: list) -> str:
        """
        Submit a generation job to the Gradio API.

        Args:
            params: Flat parameter list [model_type, param1, param2, ...]

        Returns:
            event_id string for polling progress.
        """
        if not self._endpoint:
            self.discover_endpoint()

        # Gradio 5.x call format
        call_url = f"{self.server}/gradio_api/call{self._endpoint}"
        self.log(f"Submitting job to {call_url}")

        payload = {"data": params}
        if self.verbose:
            # Truncate image data for logging
            log_params = []
            for p in params:
                if isinstance(p, (bytes, bytearray)):
                    log_params.append(f"<{len(p)} bytes>")
                else:
                    log_params.append(repr(p)[:80])
            self.log(f"Params: [{', '.join(log_params[:5])}...] ({len(params)} total)")

        r = self._session.post(call_url, json=payload, timeout=30)
        if r.status_code == 422:
            # Parameter mismatch — try re-discovery with actual param count
            self.log(f"422 error — re-discovering endpoint with {len(params)} params")
            self._endpoint = None
            self.discover_endpoint(lora_count=max(0, len(params) - 1 - len(PARAMETER_META) - 1))  # -1 for lora_weights_dict
            call_url = f"{self.server}/gradio_api/call{self._endpoint}"
            r = self._session.post(call_url, json=payload, timeout=30)

        if r.status_code != 200:
            raise RuntimeError(
                f"Job submission failed (HTTP {r.status_code}): {r.text[:500]}"
            )

        result = r.json()
        event_id = result.get("event_id")
        if not event_id:
            raise RuntimeError(f"No event_id in response: {result}")

        self.log(f"Job submitted. Event ID: {event_id}")
        return event_id

    # ── Job Monitoring via SSE ──

    def poll_job(
        self, event_id: str, timeout: float = 3600, poll_interval: float = 5.0
    ) -> Dict[str, Any]:
        """
        Poll a job's event stream until completion.

        Uses Server-Sent Events (SSE) from the Gradio 5 API.

        Args:
            event_id: The event_id from submit_job().
            timeout: Max seconds to wait.
            poll_interval: Seconds between poll attempts (SSE reconnect).

        Returns:
            Dict with outputs from the final event, or {'event_id': event_id, 'status': 'timeout'}.
        """
        if not self._endpoint:
            raise RuntimeError("No endpoint discovered. Call discover_endpoint() first.")

        sse_url = f"{self.server}/gradio_api/call{self._endpoint}/{event_id}"
        self.log(f"Polling: {sse_url}")

        start_time = time.time()
        last_data = {"status": "pending"}
        consecutive_errors = 0

        while time.time() - start_time < timeout:
            try:
                with self._session.get(sse_url, stream=True, timeout=max(10, poll_interval)) as r:
                    if r.status_code != 200:
                        self.log(f"SSE poll returned HTTP {r.status_code}")
                        time.sleep(poll_interval)
                        consecutive_errors += 1
                        if consecutive_errors > 5:
                            break
                        continue

                    consecutive_errors = 0
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            try:
                                data = json.loads(data_str)
                                last_data = data

                                # Check for completion
                                output = data.get("output", {})
                                if data.get("success") is True or "complete" in data.get("event", ""):
                                    self.log("Job completed!")
                                    return {
                                        "event_id": event_id,
                                        "status": "completed",
                                        "output": output.get("data", []),
                                        "duration": time.time() - start_time,
                                    }

                                # Progress update
                                progress = data.get("progress", {})
                                if progress:
                                    pct = progress.get("percentage", 0)
                                    desc = progress.get("desc", "")
                                    self.log(f"Progress: {pct:.0f}% — {desc}")

                            except json.JSONDecodeError:
                                pass

                        elif line.startswith("event: complete"):
                            self.log("Job completed!")
                            return {
                                "event_id": event_id,
                                "status": "completed",
                                "output": last_data.get("output", {}).get("data", [])
                                if isinstance(last_data, dict) else [],
                                "duration": time.time() - start_time,
                            }

            except requests.RequestException as e:
                self.log(f"SSE connection error: {e}")
                consecutive_errors += 1
                if consecutive_errors > 5:
                    break
                time.sleep(poll_interval)

        elapsed = time.time() - start_time
        self.log(f"Polling timed out after {elapsed:.0f}s")
        return {"event_id": event_id, "status": "timeout", "duration": elapsed}

    # ── Result Download ──

    def extract_job_id(self, output_data: list) -> Optional[str]:
        """
        Extract the job_id from the API response output data.

        The process() function returns:
            [None, job_id, None, '', 'Job added to queue...', ..., ...]
        So job_id is at index 1.
        """
        if isinstance(output_data, list) and len(output_data) > 1:
            jid = output_data[1]
            if jid:
                return str(jid)
        return None

    def wait_for_output_file(
        self,
        job_id: str,
        output_dir: str,
        timeout: float = 3600,
        poll_interval: float = 5.0,
    ) -> Optional[str]:
        """
        Watch the output directory for the result video file.

        The server saves videos to the output_dir with filenames like:
            {timestamp}_{job_id_prefix}_1.mp4

        Args:
            job_id: The job ID returned from job submission.
            output_dir: Directory to watch.
            timeout: Max seconds to wait.
            poll_interval: Seconds between checks.

        Returns:
            Path to the downloaded result video, or None if timed out.
        """
        output_dir = output_dir or self._output_dir or self.discover_output_dir()
        job_prefix = job_id[:8] if job_id else ""
        start_time = time.time()
        known_files = set(os.listdir(output_dir))

        self.log(f"Watching {output_dir} for result video (prefix: {job_prefix})")

        while time.time() - start_time < timeout:
            current_files = set(os.listdir(output_dir))
            new_files = current_files - known_files

            # Check for mp4 files matching our job
            mp4_files = sorted(
                [f for f in new_files if f.endswith(".mp4") and (not job_prefix or job_prefix in f)],
                key=lambda f: os.path.getmtime(os.path.join(output_dir, f)),
                reverse=True,
            )

            if mp4_files:
                latest = mp4_files[0]
                result_path = os.path.join(output_dir, latest)
                # Ensure file is fully written (wait for stable size)
                stable_checks = 0
                last_size = -1
                while stable_checks < 3 and time.time() - start_time < timeout:
                    try:
                        current_size = os.path.getsize(result_path)
                        if current_size == last_size and current_size > 0:
                            stable_checks += 1
                        else:
                            stable_checks = 0
                            last_size = current_size
                        if stable_checks < 3:
                            time.sleep(1)
                    except OSError:
                        time.sleep(1)
                self.log(f"Found result video: {latest} ({last_size:,} bytes)")
                return result_path

            # Also check if ALL files (not just new) match the job_id
            for f in os.listdir(output_dir):
                if f.endswith(".mp4") and job_prefix and job_prefix in f:
                    full_path = os.path.join(output_dir, f)
                    if full_path not in [os.path.join(output_dir, nf) for nf in new_files]:
                        # Already counted as known, but might be older auto-loaded result
                        pass

            time.sleep(poll_interval)

        self.log(f"Timed out waiting for output file (job_id={job_id})")
        return None

    def copy_result(self, source_path: str, output_dir: str, job_id: Optional[str] = None) -> str:
        """
        Copy the result video to a local output directory with a clean filename.

        Args:
            source_path: Path to the result video on the server.
            output_dir: Local directory to copy to.
            job_id: Optional job ID for naming the output file.

        Returns:
            Path to the copied file.
        """
        os.makedirs(output_dir, exist_ok=True)

        ext = Path(source_path).suffix
        if job_id:
            dest_name = f"framepack_{job_id[:8]}{ext}"
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dest_name = f"framepack_{timestamp}{ext}"

        dest_path = os.path.join(output_dir, dest_name)

        # If on the same machine, copy directly
        shutil.copy2(source_path, dest_path)
        self.log(f"Copied result to: {dest_path}")
        return dest_path

    # ── Full Run ──

    def run(
        self,
        model_type: str = "Original",
        input_image: Optional[str] = None,
        input_video: Optional[str] = None,
        end_frame_image: Optional[str] = None,
        prompt: str = "[1s: The person waves hello]",
        output_dir: str = "./automate_outputs",
        timeout: float = 3600,
        poll_interval: float = 5.0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Full pipeline: upload → submit → poll → download.

        Returns a dict with keys:
            success (bool), job_id (str|None), output_path (str|None),
            event_id (str|None), duration (float), error (str|None)
        """
        result: Dict[str, Any] = {
            "success": False,
            "job_id": None,
            "output_path": None,
            "event_id": None,
            "duration": 0.0,
            "error": None,
        }
        start_time = time.time()

        try:
            # 1. Check server
            if not self.check_server():
                raise RuntimeError(f"Cannot reach server at {self.server}")

            # 2. Discover server output dir
            server_output_dir = self.discover_output_dir()

            # 3. Upload files
            uploaded_image = None
            if input_image:
                uploaded_image = self.upload_file(input_image)

            uploaded_video = None
            if input_video:
                uploaded_video = self.upload_file(input_video)

            uploaded_end_frame = None
            if end_frame_image:
                uploaded_end_frame = self.upload_file(end_frame_image)

            # 4. Build parameter payload
            params_dict = {
                "input_image": uploaded_image,
                "input_video": uploaded_video,
                "end_frame_image": uploaded_end_frame,
                "end_frame_strength": kwargs.get("end_frame_strength", 1.0),
                "prompt": prompt,
                "n_prompt": kwargs.get("n_prompt", ""),
                "seed": kwargs.get("seed", 2500),
                "randomize_seed": kwargs.get("randomize_seed", False),
                "total_second_length": kwargs.get("total_second_length", 6),
                "latent_window_size": kwargs.get("latent_window_size", 9),
                "steps": kwargs.get("steps", 25),
                "cfg": kwargs.get("cfg", 1.0),
                "gs": kwargs.get("gs", 10.0),
                "rs": kwargs.get("rs", 0.0),
                "cache_type": kwargs.get("cache_type", "MagCache"),
                "teacache_num_steps": kwargs.get("teacache_num_steps", 25),
                "teacache_rel_l1_thresh": kwargs.get("teacache_rel_l1_thresh", 0.15),
                "magcache_threshold": kwargs.get("magcache_threshold", 0.1),
                "magcache_max_consecutive_skips": kwargs.get("magcache_max_consecutive_skips", 2),
                "magcache_retention_ratio": kwargs.get("magcache_retention_ratio", 0.25),
                "blend_sections": kwargs.get("blend_sections", 4),
                "latent_type": kwargs.get("latent_type", "Noise"),
                "clean_up_videos": kwargs.get("clean_up_videos", True),
                "selected_loras": kwargs.get("selected_loras", []),
                "resolutionW": kwargs.get("resolutionW", 640),
                "resolutionH": kwargs.get("resolutionH", 640),
                "combine_with_source": kwargs.get("combine_with_source", True),
                "num_cleaned_frames": kwargs.get("num_cleaned_frames", 5),
                "lora_names_states": kwargs.get("lora_names_states", []),
            }

            lora_weights_dict = kwargs.get("lora_weights_dict", None)
            params_list = build_params_list(model_type, params_dict, lora_weights_dict)

            # 5. Submit job
            try:
                event_id = self.submit_job(params_list)
            except RuntimeError as e:
                # Retry with endpoint re-discovery using exact param count
                self.log(f"Initial submit failed ({e}). Re-discovering endpoint...")
                self._endpoint = None
                self.discover_endpoint()
                event_id = self.submit_job(params_list)

            result["event_id"] = event_id

            # 6. Poll for completion
            poll_result = self.poll_job(event_id, timeout=timeout, poll_interval=poll_interval)

            if poll_result.get("status") != "completed":
                # Even if the SSE polling didn't report completion cleanly,
                # the job may still be running. Try to find the output file.
                self.log("SSE did not report completion. Trying file-watch fallback.")
                job_id_from_output = None
            else:
                job_id_from_output = self.extract_job_id(poll_result.get("output", []))

            # 7. Wait for the output file
            job_id = job_id_from_output
            if not job_id:
                # Try to extract from the output data differently
                output = poll_result.get("output", [])
                if isinstance(output, list) and len(output) > 1:
                    job_id = str(output[1]) if output[1] else None

            result["job_id"] = job_id

            output_video_path = self.wait_for_output_file(
                job_id or event_id,
                server_output_dir,
                timeout=timeout - (time.time() - start_time) if timeout > 0 else 0,
                poll_interval=poll_interval,
            )

            # 8. Copy result
            if output_video_path:
                local_path = self.copy_result(output_video_path, output_dir, job_id=job_id)
                result["output_path"] = local_path
                result["success"] = True
            else:
                self.log("No output video found within timeout.")
                result["error"] = "No output video found within timeout period."

        except Exception as e:
            self.log(f"Error: {e}")
            result["error"] = str(e)
            import traceback
            traceback.print_exc()

        result["duration"] = time.time() - start_time
        return result


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="FramePack Studio Automation — programmatically generate videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic image-to-video
  python automate.py --input-image ~/cat.png --prompt "[1s: Cat waves hello]"

  # Video-to-video with custom params
  python automate.py --input-video input.mp4 --model-type Video \\
      --prompt "Make it snowy" --steps 30 --seed 42 --verbose

  # With config file
  python automate.py --config preset.json --input-image photo.png

  # Just submit and print job ID (non-blocking)
  python automate.py --input-image photo.png --no-wait --verbose
        """,
    )

    # ── Server ──
    parser.add_argument("--server", default="http://localhost:7860",
                        help="FramePack Studio server URL (default: http://localhost:7860)")

    # ── Config ──
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file with default parameters")

    # ── Model ──
    parser.add_argument("--model-type", choices=MODEL_TYPES, default="Original",
                        help="Generation model type (default: Original)")

    # ── Input files ──
    parser.add_argument("--input-image", type=str, default=None,
                        help="Path to input image file")
    parser.add_argument("--input-video", type=str, default=None,
                        help="Path to input video file (for Video model types)")
    parser.add_argument("--end-frame", type=str, default=None,
                        help="Path to end frame image (for Endframe model types)")

    # ── Prompt ──
    parser.add_argument("--prompt", type=str,
                        default="[1s: The person waves hello]",
                        help="Generation prompt with optional timestamps")
    parser.add_argument("--negative-prompt", type=str, default="",
                        help="Negative prompt")

    # ── Basic params ──
    parser.add_argument("--seed", type=int, default=2500,
                        help="Random seed (default: 2500)")
    parser.add_argument("--randomize-seed", action="store_true",
                        help="Randomize seed for each job")
    parser.add_argument("--duration", type=float, default=6,
                        help="Video length in seconds (default: 6)")
    parser.add_argument("--steps", type=int, default=25,
                        help="Diffusion steps (default: 25, range: 1-100)")
    parser.add_argument("--width", type=int, default=640,
                        help="Output width (default: 640, step: 32)")
    parser.add_argument("--height", type=int, default=640,
                        help="Output height (default: 640, step: 32)")

    # ── Advanced params ──
    parser.add_argument("--cfg", type=float, default=1.0,
                        help="CFG scale (default: 1.0, range: 1.0-3.0)")
    parser.add_argument("--gs", type=float, default=10.0,
                        help="Distilled CFG scale (default: 10.0, range: 1.0-32.0)")
    parser.add_argument("--rs", type=float, default=0.0,
                        help="CFG re-scale (default: 0.0, range: 0.0-1.0)")
    parser.add_argument("--latent-window", type=int, default=9,
                        help="Latent window size (default: 9, range: 1-33)")
    parser.add_argument("--blend-sections", type=int, default=4,
                        help="Prompt blend sections (default: 4, range: 0-10)")
    parser.add_argument("--latent-type", choices=LATENT_TYPES, default="Noise",
                        help="Latent image type (default: Noise)")

    # ── Cache ──
    parser.add_argument("--cache-type", choices=CACHE_TYPES, default="MagCache",
                        help="Caching strategy (default: MagCache)")
    parser.add_argument("--teacache-num-steps", type=int, default=25,
                        help="TeaCache steps (default: 25)")
    parser.add_argument("--teacache-thresh", type=float, default=0.15,
                        help="TeaCache rel_l1 threshold (default: 0.15)")
    parser.add_argument("--magcache-threshold", type=float, default=0.1,
                        help="MagCache threshold (default: 0.1)")
    parser.add_argument("--magcache-max-skips", type=int, default=2,
                        help="MagCache max consecutive skips (default: 2)")
    parser.add_argument("--magcache-retention", type=float, default=0.25,
                        help="MagCache retention ratio (default: 0.25)")

    # ── LoRA ──
    parser.add_argument("--loras", type=str, default=None,
                        help="Comma-separated list of LoRA names to use")
    parser.add_argument("--lora-values", type=str, default=None,
                        help="Comma-separated LoRA weight values (must match --loras count)")

    # ── Video-specific ──
    parser.add_argument("--combine-with-source", action="store_true", default=True,
                        help="Combine output with source video (Video models)")
    parser.add_argument("--no-combine", action="store_false", dest="combine_with_source",
                        help="Don't combine output with source video")
    parser.add_argument("--num-cleaned-frames", type=int, default=5,
                        help="Number of context frames for video models (default: 5)")

    # ── Output ──
    parser.add_argument("--output-dir", type=str, default="./automate_outputs",
                        help="Directory to save result videos (default: ./automate_outputs)")

    # ── Polling ──
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Max seconds to wait for completion (default: 3600)")
    parser.add_argument("--poll-interval", type=float, default=5.0,
                        help="Seconds between status polls (default: 5.0)")

    # ── Mode ──
    parser.add_argument("--no-wait", action="store_true",
                        help="Submit job and print event_id without waiting for result")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    return parser


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def build_params_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Build a parameter dict from parsed CLI args, with config file overrides."""
    params: Dict[str, Any] = {}

    # Load config file first (as base)
    if args.config:
        config_data = load_config(args.config)
        params.update(config_data)

    # Map CLI args -> parameter names (CLI overrides config)
    cli_to_param = {
        "model_type": "model_type",
        "input_image": "input_image",
        "input_video": "input_video",
        "end_frame": "end_frame_image",
        "prompt": "prompt",
        "negative_prompt": "n_prompt",
        "seed": "seed",
        "randomize_seed": "randomize_seed",
        "duration": "total_second_length",
        "steps": "steps",
        "cfg": "cfg",
        "gs": "gs",
        "rs": "rs",
        "latent_window": "latent_window_size",
        "blend_sections": "blend_sections",
        "latent_type": "latent_type",
        "cache_type": "cache_type",
        "teacache_num_steps": "teacache_num_steps",
        "teacache_thresh": "teacache_rel_l1_thresh",
        "magcache_threshold": "magcache_threshold",
        "magcache_max_skips": "magcache_max_consecutive_skips",
        "magcache_retention": "magcache_retention_ratio",
        "width": "resolutionW",
        "height": "resolutionH",
        "combine_with_source": "combine_with_source",
        "num_cleaned_frames": "num_cleaned_frames",
    }

    for cli_name, param_name in cli_to_param.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            params[param_name] = val

    # Handle LoRA params
    if args.loras:
        lora_names = [n.strip() for n in args.loras.split(",") if n.strip()]
        params["selected_loras"] = lora_names
        params["lora_names_states"] = lora_names

        if args.lora_values:
            values = [float(v.strip()) for v in args.lora_values.split(",") if v.strip()]
            if len(values) != len(lora_names):
                raise ValueError(
                    f"LoRA value count ({len(values)}) must match LoRA count ({len(lora_names)})"
                )
            params["lora_weights_dict"] = dict(zip(lora_names, values))
        else:
            params["lora_weights_dict"] = {name: 1.0 for name in lora_names}

    # Ensure no-wait disables long timeout
    if args.no_wait:
        params["_no_wait"] = True

    return params


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Build parameters from args + config
    params = build_params_from_args(args)

    # Create client
    client = FramePackClient(server_url=args.server, verbose=args.verbose)

    no_wait = params.pop("_no_wait", False)

    if no_wait:
        # ── Non-blocking mode: just submit and print event_id ──
        client.log("Non-blocking mode: submitting and printing event_id only")

        if not client.check_server():
            print("ERROR: Cannot reach server", file=sys.stderr)
            sys.exit(1)

        client.discover_output_dir()

        # Upload files
        uploaded_image = None
        if params.get("input_image"):
            uploaded_image = client.upload_file(params["input_image"])
        uploaded_video = None
        if params.get("input_video"):
            uploaded_video = client.upload_file(params["input_video"])
        uploaded_end_frame = None
        if params.get("end_frame_image"):
            uploaded_end_frame = client.upload_file(params["end_frame_image"])

        params_dict = {
            "input_image": uploaded_image,
            "input_video": uploaded_video,
            "end_frame_image": uploaded_end_frame,
        }
        for k in PARAMETER_META:
            if k not in ("input_image", "input_video", "end_frame_image") and k in params:
                params_dict[k] = params[k]

        lora_weights_dict = params.get("lora_weights_dict")
        params_list = build_params_list(
            params.get("model_type", "Original"),
            params_dict,
            lora_weights_dict,
        )

        try:
            event_id = client.submit_job(params_list)
        except RuntimeError as e:
            client._endpoint = None
            client.discover_endpoint()
            event_id = client.submit_job(params_list)

        print(json.dumps({
            "status": "submitted",
            "event_id": event_id,
            "server": args.server,
        }))

    else:
        # ── Blocking mode: full pipeline ──
        # Extract model_type separately (it's a CLI arg, not in params dict)
        model_type = params.pop("model_type", args.model_type or "Original")

        result = client.run(
            model_type=model_type,
            output_dir=args.output_dir,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            **params,
        )

        # Print machine-readable JSON to stdout
        print(json.dumps(result, indent=2, default=str))

        if result.get("success"):
            print(f"\n✅ Video saved to: {result['output_path']}", file=sys.stderr)
            sys.exit(0)
        else:
            print(f"\n❌ Failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
