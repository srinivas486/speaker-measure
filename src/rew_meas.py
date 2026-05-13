#!/usr/bin/env python3
"""
REW Sweep-File Measurement Workflow — speaker-measure

Designed for: OCA / AudioControl sweep file packs (.mlp per speaker + reference .wav)

Prerequisites:
  - REW running on this Windows machine with -api flag:
      C:\\Program Files\\REW\\roomeqwizard.exe -api
  - sweep_folder contains:
      - One .wav file (no channel label in filename) = sweep reference for REW
      - One .mlp per speaker (FL.mlp, FR.mlp, C.mlp, SW1.mlp, SW2.mlp, etc.)
  - AVR in Dolby Atmos mode
  - UMIK-1 or calibrated mic connected

Workflow:
  1. Import sweep.wav into REW as the measurement reference (no channel label)
  2. For each .mlp file in folder:
       a. Parse channel label from filename (FL, FR, SW1, etc.)
       b. If subwoofer: show switching prompt, wait for Enter
       c. Play .mlp audio on the corresponding HDMI channel
       d. Record mic response
       e. Import captured WAV into REW → deconvolves → saves measurement
       f. Name in REW: FL0, FL1, FR0, FR1… SW10, SW11, SW20, SW21…
  3. Iterate for multiple measurements of same speaker (append numeric suffix)

Usage:
  python rew_meas.py "C:\path\to\sweep_folder" [--device-playback ID] [--device-capture ID]
  python rew_meas.py "C:\path\to\sweep_folder" --interactive

For subwoofers: the script detects SW1, SW2 etc. filenames and prompts you to
switch power to the relevant subwoofer before each sub measurement.

Requires: numpy, sounddevice, soundfile, requests (or urllib)
"""

import http.client
import json
import logging
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rew_meas")


# ── REW API Client ────────────────────────────────────────────────────────────

REW_HOST = "127.0.0.1"
REW_PORT = 4735


class RewApiError(Exception):
    """Raised when REW API call fails."""
    pass


class RewApiClient:
    def __init__(self, host: str = REW_HOST, port: int = REW_PORT):
        self.host = host
        self.port = port

    def is_available(self) -> bool:
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=3.0)
            conn.request("GET", "/")
            resp = conn.getresponse()
            conn.close()
            return resp.status == 200
        except Exception:
            return False

    def connect(self) -> None:
        if not self.is_available():
            raise RewApiError(
                f"REW not available at {self.host}:{self.port}.\n"
                "Start REW with: C:\\Program Files\\REW\\roomeqwizard.exe -api"
            )
        logger.info(f"Connected to REW at {self.host}:{self.port}")

    def _get(self, path: str) -> dict:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10.0)
        conn.request("GET", path)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        if resp.status != 200:
            raise RewApiError(f"GET {path} → {resp.status}: {data[:200]}")
        return json.loads(data) if data else {}

    def _post(self, path: str, body: Optional[dict] = None) -> dict:
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30.0)
        headers = {"Content-Type": "application/json"} if body else {}
        body_bytes = json.dumps(body).encode("utf-8") if body else b""
        conn.request("POST", path, body=body_bytes, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        if resp.status not in (200, 202):
            raise RewApiError(f"POST {path} → {resp.status}: {data[:200]}")
        return json.loads(data) if data else {}

    def clear_all_measurements(self) -> dict:
        return self._post("/measurements/clear", {})

    def import_impulse_response(
        self,
        wav_path: Path,
        measurement_name: str,
        measurement_notes: str = "",
    ) -> dict:
        """Import recorded WAV into REW. REW deconvolves against the loaded reference."""
        if not wav_path.exists():
            raise RewApiError(f"WAV not found: {wav_path}")

        try:
            audio_data, samplerate = sf.read(str(wav_path), dtype="float32")
        except Exception as e:
            raise RewApiError(f"Failed to read WAV {wav_path}: {e}")

        # Mono → stereo
        if audio_data.ndim == 1:
            audio_data = np.stack([audio_data, audio_data], axis=1)

        # Pack big-endian float32 stereo interleaved
        packed = struct.pack(f">{len(audio_data) * 2}f", *audio_data.T.flat)
        b64_data = __import__("base64").b64encode(packed).decode("ascii")

        payload = {
            "name": measurement_name,
            "notes": measurement_notes,
            "data": b64_data,
            "sampleRate": int(samplerate),
            "channels": 2,
        }

        logger.info(f"  → REW import: {measurement_name}")
        result = self._post("/measurements/importimpulseresponse", payload)
        return result


# ── Audio helpers ─────────────────────────────────────────────────────────────

def decode_truehd_mlp(mlp_path: Path) -> tuple[np.ndarray, int]:
    """Decode TrueHD MLP (.mpl) to stereo float32 PCM via ffmpeg.

    Returns (stereo_data, sample_rate=48000).
    First 2 channels extracted (remaining 6 ch are always silent in these files).
    """
    result = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(mlp_path),
        "-map", "0:a:0",
        "-af", "aformat=sample_fmts=fltp:channel_layouts=stereo",
        "-ar", "48000",
        "-f", "f32le", "pipe:1",
    ], capture_output=True)

    if result.returncode != 0:
        raise RewApiError(f"ffmpeg decode failed for {mlp_path.name}: "
                          + result.stderr.decode("utf-8", errors="replace").strip())

    raw = result.stdout
    n_floats = len(raw) // 4
    floats = np.frombuffer(raw, dtype=np.float32).copy()
    n_frames = n_floats // 2
    return floats.reshape((n_frames, 2)).astype(np.float32), 48000


def read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Read any supported audio file. .mpl → ffmpeg; others → soundfile."""
    if path.suffix.lower() == ".mpl":
        return decode_truehd_mlp(path)
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)
    return data.astype(np.float32), sr


# ── Channel parsing ─────────────────────────────────────────────────────────────

# Map filename stems to standard channel labels
_CHANNEL_ALIASES = {
    "FL": "FL", "FR": "FR",
    "C": "C", "CENTER": "C", "CENTRE": "C",
    "BL": "BL", "BR": "BR",
    "SL": "SL", "SR": "SR",
    "SRL": "SRL", "SRR": "SRR",
    "RL": "RL", "RR": "RR",
    "TFL": "TFL", "TFR": "TFR",
    "FTL": "FTL", "FTR": "FTR",
    "FHL": "FHL", "FHR": "FHR",
    "FVL": "FVL", "FVR": "FVR",
    "SW1": "SW1", "SW2": "SW2", "SW3": "SW3", "SW4": "SW4",
    "SUB1": "SW1", "SUB2": "SW2", "SUB3": "SW3", "SUB4": "SW4",
    "SUB": "SW1",
    "LFE": "LFE",
}

# Patterns checked after direct match
_CHANNEL_PATTERNS = [
    ("FRONTLEFT", "FL"), ("FRONTRIGHT", "FR"),
    ("FRONT_LEFT", "FL"), ("FRONT_RIGHT", "FR"),
    ("CENTER", "C"), ("CENTRE", "C"),
    ("BACKLEFT", "BL"), ("BACKRIGHT", "BR"),
    ("SURROUNDLEFT", "SL"), ("SURROUNDRIGHT", "SR"),
    ("REARLEFT", "RL"), ("REARRIGHT", "RR"),
    ("FRONTTOPLEFT", "FTL"), ("FRONTTOPRIGHT", "FTR"),
    ("TOPFRONTLEFT", "TFL"), ("TOPFRONTRIGHT", "TFR"),
    ("FRONTHEIGHTLEFT", "FHL"), ("FRONTHEIGHTRIGHT", "FHR"),
    ("FRONT_HEIGHT_LEFT", "FHL"), ("FRONT_HEIGHT_RIGHT", "FHR"),
    ("SUBWOOFER1", "SW1"), ("SUBWOOFER2", "SW2"),
    ("SUBWOOFER3", "SW3"), ("SUBWOOFER4", "SW4"),
]


def parse_channel_id(filename: str) -> Optional[str]:
    """Map a filename like 'FL.mlp' or 'SW1.mlp' to a channel label (e.g. 'FL', 'SW1').

    Returns None if the file is the sweep reference (.wav).
    """
    stem = Path(filename).stem.upper().replace(" ", "").replace("-", "").replace("_", "")
    if stem in _CHANNEL_ALIASES:
        return _CHANNEL_ALIASES[stem]
    for pattern, label in _CHANNEL_PATTERNS:
        if pattern in stem:
            return label
    return None


def is_subwoofer(channel_id: str) -> bool:
    return channel_id.upper().startswith("SW") or channel_id.upper() == "LFE"


def sub_index(channel_id: str) -> int:
    """Return sub number from 'SW1' → 1, 'SW2' → 2, etc. LFE → 0."""
    if channel_id.upper() == "LFE":
        return 0
    try:
        return int(channel_id[-1])
    except (IndexError, ValueError):
        return 0


# ── HDMI channel detection ──────────────────────────────────────────────────────

def find_hdmi_device() -> Optional[tuple[int, str]]:
    """Find HDMI/AVR playback device. Returns (device_index, device_name)."""
    devices = sd.query_devices()
    for dev in devices:
        if dev.get("max_output_channels", 0) >= 8:
            name = dev["name"]
            # Skip built-in, speakers, headphone
            skip = any(s in name.upper() for s in ["SPEAKER", "HEADPHONE", "REALTEK",
                                                     "BUILT-IN", "HDMI0", "DISPLAY"])
            if not skip and "HDMI" in name.upper():
                return dev["index"], name
    # Fallback: any device with >= 8 channels
    for dev in devices:
        if dev.get("max_output_channels", 0) >= 8:
            return dev["index"], dev["name"]
    return None


def detect_hdmi_channels(pb_index: int) -> list[tuple[str, int]]:
    """Get list of (channel_label, hdmi_ch_index) for the playback device.

    Uses HDMIControl / AVR Telnet if available, otherwise falls back to
    channel-count-based guessing for common layouts (9.4.6, 7.1.4, 5.1.2, etc.)
    """
    try:
        from hdmi_channel_detector import HdmiChannelDetector, ChannelInfo
        detector = HdmiChannelDetector()
        info = detector.detect_channels(pb_index)
        return [(ch.label, ch.index) for ch in info.channels]
    except Exception:
        pass

    # Fallback: infer from channel count
    dev = sd.query_device_info(pb_index)
    n = dev.get("max_output_channels", 2)
    logger.warning(f"HDMI auto-detect failed, inferring {n}-channel layout from device name")

    # Denon/Marantz X3800H 9.4.6 layout
    layouts = {
        16: [("FL",0),("C",1),("FR",2),("SRA",3),("SLA",4),("FDR",5),
             ("SDL",6),("SDR",7),("FDL",8),("SW1",9),("SW2",10),
             ("FDL2",11),("SDL2",12),("TFL",13),("TFR",14),("TVL",15)],
        14: [("FL",0),("C",1),("FR",2),("SRL",3),("SRR",4),("RL",5),("RR",6),
             ("FTL",7),("FTR",8),("TFL",9),("TFR",10),("SW1",11),("SW2",12),("LFE",13)],
        12: [("FL",0),("C",1),("FR",2),("SRL",3),("SRR",4),("RL",5),("RR",6),
             ("FTL",7),("FTR",8),("TFL",9),("TFR",10),("SW1",11),("LFE",12)],
        10: [("FL",0),("C",1),("FR",2),("BL",3),("BR",4),("SL",5),("SR",6),
             ("TFL",7),("TFR",8),("SW1",9),("LFE",10)],
        8:  [("FL",0),("C",1),("FR",2),("BL",3),("BR",4),("SL",5),("SR",6),("LFE",7)],
        6:  [("FL",0),("C",1),("FR",2),("BL",3),("BR",4),("LFE",5)],
    }
    return layouts.get(n, [(f"CH{i}", i) for i in range(n)])


def get_hdmi_output_index(channel_id: str, hdmi_channels: list[tuple[str, int]]) -> int:
    """Find HDMI output channel index for a given channel label."""
    ch_id_upper = channel_id.upper()
    for label, idx in hdmi_channels:
        if label.upper() == ch_id_upper:
            return idx
        if label.upper().startswith(ch_id_upper):
            return idx
    # fallback for SW1→SW2 etc.
    logger.warning(f"HDMI output for '{channel_id}' not found in {hdmi_channels}, using 0")
    return 0


# ── AVR Telnet control (optional — for subwoofer mode display) ──────────────────

def try_avr_telnet_connect(host: str = "192.168.1.15", port: int = 23) -> Optional:
    """Attempt AVR Telnet connection. Returns socket on success."""
    try:
        import telnetlib
        tn = telnetlib.Telnet(host, port, timeout=3)
        return tn
    except Exception:
        return None


def avr_set_subwoofer_mode(tn, mode: str = "SW1") -> None:
    """Tell AVR which subwoofer is active via Telnet (if supported)."""
    # Common Denon/Marantz: "PSSWR OFF" / "PSSWR ON" or channel-based
    try:
        tn.write(f"PSSWR OFF\r\n".encode("ascii"))
        time.sleep(0.3)
        tn.write(f"PSSWR ON\r\n".encode("ascii"))
        time.sleep(0.3)
    except Exception as e:
        logger.debug(f"AVR Telnet: {e}")


# ── Core measurement ─────────────────────────────────────────────────────────────

def play_and_capture(
    audio_data: np.ndarray,
    pb_index: int,
    cap_index: int,
    hdmi_out_index: int,
    total_output_channels: int,
    sample_rate: int,
    duration_sec: float,
) -> np.ndarray:
    """Play audio on a specific HDMI channel and record from mic simultaneously.

    Args:
        audio_data: Stereo float32 array (n_frames, 2)
        pb_index: Playback device index
        cap_index: Capture device index
        hdmi_out_index: HDMI output channel to route audio to
        total_output_channels: Total channels on playback device
        sample_rate: Sample rate
        duration_sec: Recording duration

    Returns:
        Recorded audio as float32 mono array.
    """
    n_samples = len(audio_data)

    # Build output: zeros everywhere except the target HDMI channel
    output = np.zeros((n_samples, total_output_channels), dtype=np.float32)
    # Use first channel of audio_data for the target output
    output[:, hdmi_out_index] = audio_data[:, 0]

    # Pad or trim to exact duration
    target_frames = int(duration_sec * sample_rate)
    if output.shape[0] < target_frames:
        output = np.pad(output, ((0, target_frames - output.shape[0]), (0, 0)))
    else:
        output = output[:target_frames]

    recorded = np.zeros((target_frames, 1), dtype=np.float32)

    def play_thread_fn():
        sd.play(output, device=pb_index, samplerate=sample_rate, blocking=True)

    t = threading.Thread(target=play_thread_fn, name=f"play-ch{hdmi_out_index}")
    t.start()
    time.sleep(0.05)  # small delay so capture starts after playback
    sd.rec(frames=target_frames, device=cap_index, samplerate=sample_rate,
           channels=1, dtype=np.float32, blocking=True)
    t.join()
    return recorded


def save_captured(captured: np.ndarray, path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), captured, sample_rate, subtype="FLOAT")
    logger.info(f"  Saved: {path.name}")


# ── Subwoofer switching prompt ──────────────────────────────────────────────────

def prompt_subwoofer_switch(target_sub: int, all_subs: list[int]) -> None:
    """Blocking prompt for user to switch subwoofers."""
    others = [f"SW{s}" for s in sorted(all_subs) if s != target_sub]
    print()
    print("=" * 60)
    print(f"  ⚠️  SUBWOOFER SWITCH REQUIRED")
    print("=" * 60)
    print(f"  Measure  : SW{target_sub}")
    print(f"  Switch ON : SW{target_sub}")
    if others:
        print(f"  Switch OFF: {', '.join(others)}")
    print("=" * 60)
    input("  Press ENTER after switching → measurement starts...\n")


# ── Main workflow ───────────────────────────────────────────────────────────────

def run_rew_measurement(
    sweep_folder: Path,
    pb_index: Optional[int] = None,
    cap_index: Optional[int] = None,
    sample_rate: int = 48000,
    sweep_duration_sec: float = 10.0,
    iteration_limit: int = 99,
) -> dict:
    """
    Run the REW sweep-file measurement workflow.

    Args:
        sweep_folder: Folder containing sweep.wav reference + per-channel .mlp files
        pb_index: Playback device index (auto-detect if None)
        cap_index: Capture device index (auto-detect if None)
        sample_rate: Audio sample rate
        sweep_duration_sec: Sweep / recording duration
        iteration_limit: Max iterations per channel

    Returns:
        dict mapping meas_name → result dict
    """
    rew = RewApiClient()

    # ── 1. Connect to REW ─────────────────────────────────────────────────────
    logger.info("Connecting to REW...")
    try:
        rew.connect()
    except RewApiError as e:
        logger.error(f"REW connection failed: {e}")
        sys.exit(1)

    # ── 2. Discover files ────────────────────────────────────────────────────
    all_files = list(sweep_folder.glob("*.wav")) + list(sweep_folder.glob("*.WAV")) \
              + list(sweep_folder.glob("*.mlp")) + list(sweep_folder.glob("*.MPL"))

    sweep_wav_path = None
    channel_files: dict[str, Path] = {}  # channel_id → .mlp path

    for f in all_files:
        cid = parse_channel_id(f.name)
        if f.suffix.lower() == ".wav" and cid is None:
            sweep_wav_path = f
        elif cid is not None:
            channel_files[cid] = f

    if not sweep_wav_path:
        logger.error("No .wav sweep reference found in folder (a file with no channel label)")
        sys.exit(1)
    if not channel_files:
        logger.error("No .mlp speaker files found in folder")
        sys.exit(1)

    logger.info(f"Reference sweep : {sweep_wav_path.name}")
    logger.info(f"Speaker channels : {sorted(channel_files.keys())}")

    # Load the sweep reference once
    logger.info("Loading sweep reference...")
    ref_data, ref_sr = read_audio_file(sweep_wav_path)
    logger.info(f"  Reference: {len(ref_data)} samples ({len(ref_data)/ref_sr:.1f}s @ {ref_sr}Hz)")

    # ── 3. Device detection ─────────────────────────────────────────────────
    # Capture
    if cap_index is None:
        caps = [d for d in sd.query_devices() if d.get("max_input_channels", 0) >= 1]
        if not caps:
            logger.error("No capture devices found")
            sys.exit(1)
        if len(caps) == 1:
            cap_index = caps[0]["index"]
        else:
            print("\n=== Capture Devices ===")
            for d in caps:
                print(f"  [{d['index']}] {d['name']}")
            cap_index = int(input("Select capture device: ").strip())

    # Playback
    if pb_index is None:
        hdmi = find_hdmi_device()
        if hdmi:
            pb_index, pb_name = hdmi
            logger.info(f"HDMI/AVR auto-detected: {pb_name} (index {pb_index})")
        else:
            pbs = [d for d in sd.query_devices() if d.get("max_output_channels", 0) >= 2]
            print("\n=== Playback Devices ===")
            for d in pbs:
                print(f"  [{d['index']}] {d['name']} ({d['max_output_channels']}ch)")
            pb_index = int(input("Select playback device: ").strip())

    dev_info = sd.query_device_info(pb_index)
    total_output_channels = dev_info.get("max_output_channels", 2)
    logger.info(f"Playback: {dev_info['name']} ({total_output_channels}ch)")
    logger.info(f"Capture  : {sd.query_device_info(cap_index)['name']}")

    # HDMI channel map
    hdmi_channels = detect_hdmi_channels(pb_index)
    logger.info(f"HDMI channels: {hdmi_channels}")

    # ── 4. Import sweep.wav as reference in REW ─────────────────────────────
    logger.info("Loading sweep as REW reference...")
    try:
        # The sweep.wav becomes the reference measurement in REW
        rew.import_impulse_response(
            sweep_wav_path,
            measurement_name="REFERENCE",
            measurement_notes=f"Sweep reference: {sweep_wav_path.name}",
        )
        logger.info("Reference loaded in REW")
    except RewApiError as e:
        logger.warning(f"Could not load sweep as reference: {e}")
        logger.warning("Continuing — measurements will use REW's default reference")

    # ── 5. Iterate through channels ──────────────────────────────────────────
    results: dict = {}
    iteration_count: dict[str, int] = {}  # channel_id → count for naming

    # Separate subs from main channels
    sub_channels = {cid: p for cid, p in channel_files.items() if is_subwoofer(cid)}
    main_channels = {cid: p for cid, p in channel_files.items() if not is_subwoofer(cid)}

    all_ordered = sorted(main_channels.items()) + sorted(sub_channels.items())

    # Unique sub indices for switching
    all_sub_indices = sorted(set(sub_index(cid) for cid in sub_channels if sub_index(cid) > 0))

    for i, (channel_id, mpl_path) in enumerate(all_ordered):
        iter_num = iteration_count.get(channel_id, 0)
        # Subwoofer naming: SW1 iter0→SW10, iter1→SW11; SW2 iter0→SW20, etc.
        if is_subwoofer(channel_id) and sub_index(channel_id) > 0:
            meas_name = f"SW{sub_index(channel_id)}{iter_num}"
        else:
            meas_name = f"{channel_id}{iter_num}"

        # Subwoofer switching prompt
        if is_subwoofer(channel_id) and sub_index(channel_id) > 0:
            prompt_subwoofer_switch(sub_index(channel_id), all_sub_indices)

        # Progress
        frac_base = 0.1 + 0.85 * (i / max(len(all_ordered) - 1, 1))
        logger.info(f"[{frac_base*100:.0f}%] {meas_name} ← {mpl_path.name}")

        try:
            # Load MLP audio
            ch_data, ch_sr = read_audio_file(mpl_path)
            if ch_sr != sample_rate:
                logger.warning(f"  MLP sample rate {ch_sr} ≠ expected {sample_rate}")

            # Get HDMI output index for this channel
            hdmi_out = get_hdmi_output_index(channel_id, hdmi_channels)

            # Play MLP and capture mic response
            captured = play_and_capture(
                ch_data, pb_index, cap_index,
                hdmi_out, total_output_channels,
                sample_rate, sweep_duration_sec,
            )

            # Save captured WAV
            captured_dir = sweep_folder / "captured"
            captured_path = captured_dir / f"{meas_name}_captured.wav"
            save_captured(captured, captured_path, sample_rate)

            # Import into REW (deconvolves against reference)
            rew.import_impulse_response(
                captured_path,
                measurement_name=meas_name,
                measurement_notes=f"Channel: {channel_id} | File: {mpl_path.name}",
            )
            results[meas_name] = {"success": True, "mlp": str(mpl_path),
                                  "captured": str(captured_path), "error": None}

        except Exception as e:
            logger.error(f"  Error: {e}")
            results[meas_name] = {"success": False, "mlp": str(mpl_path),
                                   "captured": None, "error": str(e)}

        # Increment iteration
        iteration_count[channel_id] = iter_num + 1

    logger.info("Done.")
    ok = sum(1 for r in results.values() if r["success"])
    logger.info(f"Results: {ok}/{len(results)} OK")
    return results


# ── CLI entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="REW sweep-file measurement for Atmos/truehd speaker files"
    )
    parser.add_argument("sweep_folder", type=Path, help="Folder with sweep.wav + *.mlp files")
    parser.add_argument("--device-playback", type=int, default=None,
                        help="Playback device index (HDMI/AVR)")
    parser.add_argument("--device-capture", type=int, default=None,
                        help="Capture device index (UMIK-1)")
    parser.add_argument("--sample-rate", type=int, default=48000,
                        help="Sample rate (default 48000)")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Sweep duration in seconds (default 10.0)")
    parser.add_argument("--iterate", type=int, default=99,
                        help="Max iterations per channel (default 99)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    folder = args.sweep_folder
    if not folder.exists():
        logger.error(f"Folder not found: {folder}")
        sys.exit(1)

    results = run_rew_measurement(
        sweep_folder=folder,
        pb_index=args.device_playback,
        cap_index=args.device_capture,
        sample_rate=args.sample_rate,
        sweep_duration_sec=args.duration,
        iteration_limit=args.iterate,
    )

    # Summary
    print("\n=== Results ===")
    for name, res in sorted(results.items()):
        tag = "✓" if res["success"] else "✗"
        err = f" — {res['error']}" if res["error"] else ""
        print(f"  {tag} {name}{err}")


if __name__ == "__main__":
    main()