"""
REW Measurement Workflow — Phase 3b (Revised)
speaker-measure

Measurement path using REW for deconvolution + recorded sweep files.

File types in sweep folder:
  - One *.wav file: the sweep stimulus (shared across all measurements)
  - One *.mpl file per speaker/speaker-type: lossless audio for that channel

.wav  → sweep stimulus (e.g. "Sweep.wav" or "Reference.wav")
.mpl  → lossless audio file for a specific speaker (FL, FR, C, SW1, SW2 …)

Workflow per channel:
  1. Show subwoofer switching prompt if measuring a subwoofer
     → "Switch ON SW1. Switch OFF all other subwoofers. Press Enter when ready."
  2. Play the sweep .wav on the AVR (AVR stays in Dolby Atmos mode — no switching)
  3. Record mic response on UMIK-1
  4. Import captured recording into REW for deconvolution + processing
  5. REW produces the frequency response for that channel

AVR stays in Dolby Atmos mode throughout. Subwoofer isolation is done
by the user physically switching — this app coordinates the prompts and timing.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import sounddevice as sd
import soundfile as sf

from audio_engine import AudioEngine
from hdmi_channel_detector import HdmiChannelDetector, HdmiDeviceInfo
from rew_api import RewApiClient, RewApiError

logger = logging.getLogger(__name__)


# Extension → (description, is_subwoofer_candidate)
_AUDIO_EXTENSIONS = {
    ".mpl": "Meridian Lossless Playlist (TrueHD per-channel speaker file, 8ch)",
    ".wav": "Wave audio (REW reference sweep stimulus)",
    ".flac": "FLAC lossless",
    ".aiff": "AIFF lossless",
}

# TrueHD MLP files are 8-channel. Each file has ONE active channel (the rest are silent).
# The active channel is always the first channel of the file.
# MLP files cannot be read by soundfile — use ffmpeg subprocess to decode to PCM.


def _read_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Read an audio file, returning (audio_data, sample_rate).

    For WAV/FLAC/AIFF: uses soundfile.
    For TrueHD MLP (.mpl): decodes via ffmpeg subprocess, extracts first 2 channels.
    Returns shape (n_samples, 2) stereo float32 array.
    """
    if path.suffix.lower() == ".mpl":
        return _decode_truehd_mlp(path)
    # WAV, FLAC, AIFF
    try:
        data, sr = sf.read(str(path), dtype="float32")
        # Ensure stereo (duplicate mono if needed)
        if data.ndim == 1:
            data = np.stack([data, data], axis=1)
        return data, sr
    except Exception as e:
        raise RewApiError(f"Cannot read {path.name}: {e}")


def _decode_truehd_mlp(mlp_path: Path) -> tuple[np.ndarray, int]:
    """Decode a TrueHD MLP file to stereo float32 PCM via ffmpeg.

    Returns (audio_data, sample_rate=48000).
    Only the first 2 channels are extracted (remaining 6 ch are always silent).
    """
    import subprocess

    result = subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(mlp_path),
        "-map", "0:a:0",
        "-af", "aformat=sample_fmts=fltp:channel_layouts=stereo",
        "-ar", "48000",
        "-f", "f32le", "pipe:1",
    ], capture_output=True)

    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace").strip()
        raise RewApiError(f"ffmpeg decode failed for {mlp_path.name}: {err}")

    raw = result.stdout
    n_floats = len(raw) // 4
    floats = np.frombuffer(raw, dtype=np.float32).copy()
    # Stereo interleaved: [L0,R0,L1,R1,...]
    n_frames = n_floats // 2
    stereo = floats.reshape((n_frames, 2))
    return stereo.astype(np.float32), 48000



def _guess_channel_id_from_filename(filename: str) -> Optional[str]:
    """Map a sweep-folder filename to a standard channel label.

    Handles patterns like: "FL.mpl", "SW1.mpl", "Center.mpl", "Sub2.mpl",
    "Front Left.mpl", "sweep.wav", etc.
    Returns None if the file appears to be the sweep reference.
    """
    stem = Path(filename).stem.upper().replace(" ", "").replace("-", "").replace("_", "")

    # Known channel labels
    channel_map = {
        "FL": "FL", "FR": "FR",
        "C": "C", "CENTER": "C",
        "BL": "BL", "BR": "BR",
        "SL": "SL", "SR": "SR",
        "SRL": "SRL", "SRR": "SRR",
        "RL": "RL", "RR": "RR",
        "FTL": "FTL", "FTR": "FTR",
        "TFL": "TFL", "TFR": "TFR",
        "FHL": "FHL", "FHR": "FHR",
        "FVL": "FVL", "FVR": "FVR",
        "SW1": "SW1", "SW2": "SW2",
        "SW3": "SW3", "SW4": "SW4",
        "SUB1": "SW1", "SUB2": "SW2",
        "SUB": "SW1",
        "LFE": "LFE",
    }

    # Direct label match
    if stem in channel_map:
        return channel_map[stem]

    # Partial match (e.g. "FrontLeft" → FL)
    for pattern, label in [
        ("FRONTLEFT", "FL"), ("FRONTRIGHT", "FR"),
        ("CENTER", "C"), ("CENTRE", "C"),
        ("BACKLEFT", "BL"), ("BACKRIGHT", "BR"),
        ("SURROUNDLEFT", "SL"), ("SURROUNDRIGHT", "SR"),
        ("REARLEFT", "RL"), ("REARRIGHT", "RR"),
        ("FRONTTOPLEFT", "FTL"), ("FRONTTOPRIGHT", "FTR"),
        ("TOPFRONTLEFT", "TFL"), ("TOPFRONTRIGHT", "TFR"),
        ("FRONTHEIGHTLEFT", "FHL"), ("FRONTHEIGHTRIGHT", "FHR"),
        ("FRONT_HEIGHT_LEFT", "FHL"), ("FRONT_HEIGHT_RIGHT", "FHR"),
        ("VOTEFROMSOVERHEADLEFT", "FVL"), ("VOTEFROMSOVERHEADRIGHT", "FVR"),
        ("SUBWOOFER1", "SW1"), ("SUBWOOFER2", "SW2"),
        ("SUBWOOFER3", "SW3"), ("SUBWOOFER4", "SW4"),
    ]:
        if pattern in stem:
            return label

    return None


def _is_subwoofer(channel_id: str) -> bool:
    return channel_id.upper() in ("SW1", "SW2", "SW3", "SW4", "LFE")


class RewMeasurementWorkflow:
    """REW-based measurement workflow using pre-recorded sweep files.

    Usage:
        workflow = RewMeasurementWorkflow(
            sweep_folder="/path/to/sweep_files",
            playback_device_index=4,
            capture_device_index=2,
        )
        workflow.set_progress_callback(my_callback)
        results = workflow.run()
    """

    def __init__(
        self,
        sweep_folder: str | Path,
        playback_device_index: Optional[int] = None,
        capture_device_index: Optional[int] = None,
        sample_rate: int = 48000,
        sweep_duration_sec: float = 10.0,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        self.sweep_folder = Path(sweep_folder)
        self.playback_device_index = playback_device_index
        self.capture_device_index = capture_device_index
        self.sample_rate = sample_rate
        self.sweep_duration_sec = sweep_duration_sec
        self.progress_callback = progress_callback
        self.engine = AudioEngine()
        self.rew = RewApiClient()

    # -------------------------------------------------------------------------
    # Progress reporting
    # -------------------------------------------------------------------------

    def _report(self, message: str, fraction: float) -> None:
        if self.progress_callback:
            self.progress_callback(message, fraction)
        logger.info(f"[{fraction*100:.0f}%] {message}")

    def _pause_for_subwoofer_switch(
        self,
        target_sub_index: int,
        all_sub_indices: list[int],
    ) -> None:
        """Print a blocking prompt instructing the user to switch subwoofers.

        Args:
            target_sub_index: Subwoofer to keep ON (e.g. 1 for SW1)
            all_sub_indices: All detected subwoofer indices
        """
        others = [s for s in all_sub_indices if s != target_sub_index]
        others_str = ", ".join(f"SW{s}" for s in others) if others else "none"
        target_str = f"SW{target_sub_index}"

        print()
        print("=" * 60)
        print(f"  ⚠️  SUBWOOFER SWITCH REQUIRED")
        print("=" * 60)
        print(f"  Measure : {target_str}")
        print(f"  Switch ON : {target_str}")
        print(f"  Switch OFF: {others_str}")
        print("=" * 60)
        input("  Press ENTER when ready to measure...\n")

    # -------------------------------------------------------------------------
    # Prerequisites
    # -------------------------------------------------------------------------

    def check_prerequisites(self) -> bool:
        """Verify sweep folder exists, has the right files, and REW API is up."""
        self._report("Checking prerequisites...", 0.0)

        if not self.sweep_folder.exists():
            logger.error(f"Sweep folder does not exist: {self.sweep_folder}")
            return False

        all_files = []
        for ext in _AUDIO_EXTENSIONS:
            all_files.extend(list(self.sweep_folder.glob(f"*{ext}")))
            all_files.extend(list(self.sweep_folder.glob(f"*{ext.upper()}")))

        if not all_files:
            logger.error(f"No audio files found in: {self.sweep_folder}")
            return False

        logger.info(f"Found {len(all_files)} audio files in {self.sweep_folder}")
        for f in sorted(all_files):
            logger.info(f"  {f.name}")

        if not self.rew.is_available():
            logger.error(
                "REW API not available at 127.0.0.1:4735.\n"
                "Start REW with -api argument:\n"
                "  Windows: C:\\Program Files\\REW\\roomeqwizard.exe -api\n"
                "  macOS  : open -a REW.app --args -api"
            )
            return False

        try:
            self.rew.connect()
        except RewApiError as e:
            logger.error(f"REW API connection failed: {e}")
            return False

        logger.info("Prerequisites OK")
        return True

    # -------------------------------------------------------------------------
    # Device detection
    # -------------------------------------------------------------------------

    def detect_devices(
        self,
        playback_name_fragment: Optional[str] = None,
    ) -> tuple[int, int]:
        """Auto-detect or prompt for playback (HDMI/AVR) and capture (UMIK-1) devices.

        Returns:
            (playback_device_index, capture_device_index)
        """
        self._report("Detecting audio devices...", 0.05)

        # Capture: UMIK-1 or any single mic
        umik = self.engine.find_umik1()
        if umik:
            cap_index = umik.id
            logger.info(f"UMIK-1 auto-detected: {umik.name}")
        else:
            caps = self.engine.list_capture_devices()
            if not caps:
                raise RuntimeError("No capture devices found")
            if len(caps) == 1:
                cap_index = caps[0].id
            else:
                print("\n=== Capture Devices ===")
                for dev in caps:
                    print(f"  [{dev.id}] {dev.name}")
                cap_str = input("Select capture device id: ").strip()
                cap_index = int(cap_str) if cap_str else caps[0].id

        # Playback: HDMI/AVR
        if self.playback_device_index is not None:
            pb_index = self.playback_device_index
        elif playback_name_fragment:
            dev = self.engine.find_device_by_name(playback_name_fragment)
            pb_index = dev.id if dev else None
        else:
            hdmi = self.engine.find_hdmi_audio()
            if hdmi:
                pb_index = hdmi.id
                logger.info(f"HDMI/AVR auto-detected: {hdmi.name}")
            else:
                pbs = self.engine.list_playback_devices()
                print("\n=== Playback Devices ===")
                for dev in pbs:
                    print(f"  [{dev.id}] {dev.name} ({dev.max_output_channels}ch)")
                pb_str = input("Select playback device id: ").strip()
                pb_index = int(pb_str) if pb_str else None

        if pb_index is None:
            raise RuntimeError("No playback device selected")

        return pb_index, cap_index

    # -------------------------------------------------------------------------
    # File discovery
    # -------------------------------------------------------------------------

    def discover_files(self) -> tuple[Optional[Path], dict[str, Path], list[str]]:
        """Find the sweep .wav and per-channel .mpl files.

        Returns:
            (sweep_wav_path, channel_file_map, unmatched_files)
            channel_file_map: {channel_id: path} for each .mpl with a channel label
            unmatched_files: list of files found but not matched to a channel
        """
        all_audio_files: list[Path] = []
        for ext in _AUDIO_EXTENSIONS:
            all_audio_files.extend(list(self.sweep_folder.glob(f"*{ext}")))
            all_audio_files.extend(list(self.sweep_folder.glob(f"*{ext.upper()}")))

        sweep_wav = None
        channel_map: dict[str, Path] = {}
        unmatched: list[str] = []

        for f in all_audio_files:
            cid = _guess_channel_id_from_filename(f.name)

            if f.suffix.lower() == ".wav" and cid is None:
                # This .wav is the sweep reference (no channel label matched)
                sweep_wav = f
            elif cid is not None:
                channel_map[cid] = f
            else:
                unmatched.append(f.name)

        # Sort channel map by standard ordering
        channel_order = [
            "FL", "C", "FR", "FHL", "FHR",
            "FVL", "FVR",
            "SL", "SR", "BL", "BR",
            "RL", "RR",
            "FTL", "FTR", "TFL", "TFR",
            "SW1", "SW2", "SW3", "SW4", "LFE",
        ]
        sorted_map = {}
        for key in channel_order:
            if key in channel_map:
                sorted_map[key] = channel_map[key]

        return sweep_wav, sorted_map, unmatched

    def detect_hdmi_layout(self, pb_index: int) -> HdmiDeviceInfo:
        """Detect HDMI channel layout from the playback device, or fallback."""
        self._report("Detecting HDMI channel layout...", 0.08)
        detector = HdmiChannelDetector()
        try:
            info = detector.detect_channels(pb_index)
            logger.info(f"HDMI layout: {info.layout_type}")
            return info
        except Exception as e:
            logger.warning(f"HDMI detection failed ({e}), using channel count fallback")
            return self._detect_by_channel_count(pb_index)

    def _detect_by_channel_count(self, pb_index: int) -> HdmiDeviceInfo:
        """Fallback layout detection from channel count."""
        from hdmi_channel_detector import ChannelInfo
        dev = self.engine.query_device_info(pb_index)
        n = dev.get("max_output_channels", 2)
        layout_map = {
            2:  [("FL", 0), ("FR", 1)],
            6:  [("FL", 0), ("C", 1), ("FR", 2), ("BL", 3), ("BR", 4), ("LFE", 5)],
            8:  [("FL", 0), ("C", 1), ("FR", 2), ("BL", 3), ("BR", 4),
                 ("SL", 5), ("SR", 6), ("LFE", 7)],
            10: [("FL", 0), ("C", 1), ("FR", 2), ("FHL", 3), ("FHR", 4),
                 ("BL", 5), ("BR", 6), ("SL", 7), ("SR", 8), ("LFE", 9)],
            12: [("FL", 0), ("C", 1), ("FR", 2), ("SRL", 3), ("SRR", 4),
                 ("RL", 5), ("RR", 6), ("FTL", 7), ("FTR", 8),
                 ("TFL", 9), ("TFR", 10), ("LFE", 11)],
            14: [("FL", 0), ("C", 1), ("FR", 2), ("SRL", 3), ("SRR", 4),
                 ("RL", 5), ("RR", 6), ("FTL", 7), ("FTR", 8),
                 ("TFL", 9), ("TFR", 10), ("SW1", 11), ("SW2", 12), ("LFE", 13)],
        }
        labels = layout_map.get(n, [("CH", i) for i in range(n)])
        channels = [
            ChannelInfo(
                label=lbl,
                index=idx,
                is_subwoofer=(lbl.upper() in ("LFE", "SW1", "SW2", "SW3", "SW4")),
            )
            for lbl, idx in labels
        ]
        return HdmiDeviceInfo(layout_type=f"{n}ch", channels=channels, subwoofer_count=0)

    def _get_hdmi_index(self, channel_id: str, device_info: HdmiDeviceInfo) -> int:
        """Find HDMI channel index for a given channel label."""
        for ch in device_info.channels:
            if ch.label.upper() == channel_id.upper():
                return ch.index
        # Fallback: match by first part (SW1 → SW1, etc.)
        for ch in device_info.channels:
            if channel_id.upper().startswith(ch.label.upper()):
                return ch.index
        logger.warning(f"HDMI index not found for {channel_id}, using 0")
        return 0

    # -------------------------------------------------------------------------
    # Per-channel play + record
    # -------------------------------------------------------------------------

    def _play_and_capture(
        self,
        audio_data: np.ndarray,
        pb_index: int,
        cap_index: int,
        hdmi_channel_index: int,
        n_dev_ch: int,
    ) -> np.ndarray:
        """Play audio on a specific HDMI channel and record from the mic.

        Args:
            audio_data: Stereo or mono float32 array
            pb_index: Playback device index
            cap_index: Capture device index
            hdmi_channel_index: HDMI output channel to play on
            n_dev_ch: Total number of output channels on the playback device

        Returns:
            Recorded audio as float32 mono (or stereo) array
        """
        # Build output matrix: (n_samples, n_dev_ch)
        n_samples = len(audio_data)
        output = np.zeros((n_samples, n_dev_ch), dtype=np.float32)

        if audio_data.ndim == 1:
            # Mono: duplicate to both L+R, then route to hdmi_channel_index
            audio_data = np.stack([audio_data, audio_data], axis=1)

        # Route the file's channels to the target HDMI output
        # If the file is stereo (2ch) and we need to play on HDMI index 4,
        # we put both file channels (or just the first) onto output channel 4.
        # Most sweep files are stereo — take the first channel.
        for ch_i in range(min(audio_data.shape[1], 2)):  # use up to 2 ch from file
            output[:, hdmi_channel_index] = audio_data[:, ch_i]

        frames = int(self.sweep_duration_sec * self.sample_rate)
        recorded = np.zeros((frames, 1), dtype=np.float32)

        def play_fn():
            sd.play(output, device=pb_index, samplerate=self.sample_rate, blocking=True)

        play_thread = threading.Thread(target=play_fn, name=f"play-{hdmi_channel_index}")
        play_thread.start()
        time.sleep(0.05)  # small head start so we don't miss the leading edge
        recorded = sd.rec(
            frames=frames,
            device=cap_index,
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            blocking=True,
        )
        play_thread.join()
        return recorded

    # -------------------------------------------------------------------------
    # REW import helpers
    # -------------------------------------------------------------------------

    def _import_to_rew(self, captured_wav: Path, channel_id: str) -> bool:
        """Import a captured WAV into REW as a measurement.

        REW will deconvolve the sweep from it and produce frequency response.
        """
        try:
            self.rew.import_impulse_response(
                captured_wav,
                measurement_name=channel_id,
                measurement_notes=f"REW workflow — {captured_wav.name}",
            )
            logger.info(f"REW import OK: {channel_id}")
            return True
        except RewApiError as e:
            logger.error(f"REW import failed for {channel_id}: {e}")
            return False

    # -------------------------------------------------------------------------
    # Main workflow
    # -------------------------------------------------------------------------

    def run(self) -> dict[str, dict]:
        """Run the complete REW measurement workflow.

        Returns:
            Dict mapping channel_id → result dict:
                success, sweep_file, captured_file, error
        """
        results: dict[str, dict] = {}

        # ── 1. Prerequisites ──────────────────────────────────────────
        if not self.check_prerequisites():
            return results

        # ── 2. Device detection ──────────────────────────────────────
        pb_index, cap_index = self.detect_devices()
        dev_info = self.engine.query_device_info(pb_index)
        n_dev_ch = dev_info.get("max_output_channels", 2)

        # ── 3. Discover files ─────────────────────────────────────────
        sweep_wav, channel_file_map, unmatched = self.discover_files()
        if not sweep_wav:
            logger.error("No sweep .wav file found in folder (a .wav without channel label)")
            return results
        if not channel_file_map:
            logger.error("No per-channel .mpl files found")
            return results

        logger.info(f"Sweep file  : {sweep_wav.name}")
        logger.info(f"Channel map : {list(channel_file_map.keys())}")
        if unmatched:
            logger.warning(f"Unmatched files (no channel label): {unmatched}")

        # Load the sweep stimulus once
        self._report("Loading sweep reference...", 0.1)
        sweep_data, sr = _read_audio_file(sweep_wav)
        if sr != self.sample_rate:
            logger.warning(f"Sweep sample rate {sr} ≠ {self.sample_rate}, resampling may be needed")
        logger.info(f"Sweep loaded: {len(sweep_data)} samples ({len(sweep_data)/sr:.1f}s)")

        # ── 4. HDMI layout ────────────────────────────────────────────
        hdmi_info = self.detect_hdmi_layout(pb_index)

        # ── 5. Identify subwoofers ───────────────────────────────────
        subwoofer_ids = [cid for cid in channel_file_map if _is_subwoofer(cid)]
        subwoofer_indices = sorted(set(
            int(cid.replace("SW", "").replace("LFE", "0"))
            for cid in subwoofer_ids
            if cid.upper() != "LFE"
        ))
        # SW1→1, SW2→2 etc.
        subwoofer_indices = sorted(set(
            int(cid[-1]) for cid in subwoofer_ids if cid.upper().startswith("SW")
        ))

        # ── 6. Clear REW ───────────────────────────────────────────────
        self._report("Clearing REW measurements...", 0.12)
        try:
            self.rew.clear_all_measurements()
        except RewApiError as e:
            logger.warning(f"Could not clear REW: {e}")

        # ── 7. Measure non-subwoofer channels ─────────────────────────
        non_subs = {cid: p for cid, p in channel_file_map.items() if not _is_subwoofer(cid)}
        total_main = len(non_subs) + len(subwoofer_ids)
        done_main = 0

        for i, (channel_id, mpl_path) in enumerate(sorted(non_subs.items())):
            fraction = 0.15 + 0.70 * (done_main / total_main)
            self._report(f"Measuring {channel_id} ({mpl_path.name})...", fraction)

            hdmi_idx = self._get_hdmi_index(channel_id, hdmi_info)

            try:
                # Load the per-channel .mpl file for this speaker
                ch_data, _ = _read_audio_file(mpl_path)

                # Play sweep.wav while the per-channel .mpl file content is
                # used as the basis for the recording (the .mpl is the "truth"
                # for what was played — we still sweep.wav as the stimulus
                # since it's what we generated/know the inverse for)
                captured = self._play_and_capture(
                    sweep_data,  # always play the sweep.wav
                    pb_index, cap_index, hdmi_idx, n_dev_ch,
                )

                # Save captured
                recorded_dir = self.sweep_folder / "captured"
                recorded_dir.mkdir(exist_ok=True)
                out_path = recorded_dir / f"{channel_id}_captured.wav"
                sf.write(str(out_path), captured, self.sample_rate)

                # Import into REW
                imported = self._import_to_rew(out_path, channel_id)
                results[channel_id] = {
                    "success": imported,
                    "sweep_file": str(sweep_wav),
                    "channel_file": str(mpl_path),
                    "captured_file": str(out_path),
                    "error": None if imported else "REW import failed",
                }
            except Exception as e:
                logger.error(f"Error measuring {channel_id}: {e}")
                results[channel_id] = {
                    "success": False,
                    "sweep_file": str(sweep_wav),
                    "channel_file": str(mpl_path),
                    "captured_file": None,
                    "error": str(e),
                }
            done_main += 1

        # ── 8. Measure subwoofer channels (with switching prompts) ─────
        for i, channel_id in enumerate(sorted(subwoofer_ids)):
            fraction = 0.15 + 0.70 * ((len(non_subs) + i) / total_main)
            mpl_path = channel_file_map[channel_id]

            # Extract sub index (e.g. "SW2" → 2)
            sub_idx = int(channel_id[-1]) if channel_id.upper().startswith("SW") else 0

            self._report(f"Subwoofer {channel_id}: switch prompts...", fraction)
            self._pause_for_subwoofer_switch(sub_idx, subwoofer_indices)

            self._report(f"Measuring {channel_id} ({mpl_path.name})...", fraction + 0.02)

            hdmi_idx = self._get_hdmi_index(channel_id, hdmi_info)

            try:
                captured = self._play_and_capture(
                    sweep_data,
                    pb_index, cap_index, hdmi_idx, n_dev_ch,
                )

                recorded_dir = self.sweep_folder / "captured"
                recorded_dir.mkdir(exist_ok=True)
                out_path = recorded_dir / f"{channel_id}_captured.wav"
                sf.write(str(out_path), captured, self.sample_rate)

                imported = self._import_to_rew(out_path, channel_id)
                results[channel_id] = {
                    "success": imported,
                    "sweep_file": str(sweep_wav),
                    "channel_file": str(mpl_path),
                    "captured_file": str(out_path),
                    "error": None if imported else "REW import failed",
                }
            except Exception as e:
                logger.error(f"Error measuring {channel_id}: {e}")
                results[channel_id] = {
                    "success": False,
                    "sweep_file": str(sweep_wav),
                    "channel_file": str(mpl_path),
                    "captured_file": None,
                    "error": str(e),
                }

        self._report("Measurement workflow complete", 1.0)

        ok = sum(1 for r in results.values() if r["success"])
        logger.info(f"Results: {ok}/{len(results)} channels imported into REW")
        return results

    # -------------------------------------------------------------------------
    # Interactive CLI
    # -------------------------------------------------------------------------

    def run_interactive(self) -> None:
        """Interactive CLI entry point."""
        print("\n=== REW Measurement Workflow ===")
        print(" AVR stays in Dolby Atmos mode — subwoofer switching is manual.\n")

        folder = input("Sweep folder path: ").strip()
        if not folder:
            print("No folder specified — aborting.")
            return
        self.sweep_folder = Path(folder)

        sr_str = input(f"Sample rate [{self.sample_rate}]: ").strip()
        if sr_str:
            self.sample_rate = int(sr_str)

        dur_str = input(f"Sweep duration sec [{self.sweep_duration_sec}]: ").strip()
        if dur_str:
            self.sweep_duration_sec = float(dur_str)

        results = self.run()

        print("\n=== Results ===")
        for ch_id, res in sorted(results.items()):
            tag = "✓" if res["success"] else "✗"
            err = f" — {res['error']}" if res["error"] else ""
            print(f"  {tag} {ch_id}{err}")


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python rew_measurement.py <sweep_folder> [--interactive]")
        sys.exit(1)

    interactive = "--interactive" in sys.argv
    folder = Path([a for a in sys.argv[1:] if not a.startswith("--")][0])

    wf = RewMeasurementWorkflow(
        sweep_folder=folder,
        sweep_duration_sec=10.0,
    )

    if interactive:
        wf.run_interactive()
    else:
        results = wf.run()
        ok = sum(1 for r in results.values() if r["success"])
        print(f"\n{ok}/{len(results)} channels measured successfully")
        for ch, res in results.items():
            print(f"  {ch}: {'OK' if res['success'] else res['error']}")
