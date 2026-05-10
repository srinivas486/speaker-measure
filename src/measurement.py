# Measurement Orchestrator — Phase 3
# speaker-measure
# Coordinates per-channel measurement: sweep → capture → deconvolve → export
# Handles multi-subwoofer measurement with user-guided subwoofer switching.

import numpy as np
import sounddevice as sd
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Callable, Protocol
import logging

from audio_engine import AudioEngine
from signal_processor import SignalProcessor
from mic_calibration import MicCalibration
from exporter import WavExporter, OffsetCalculator, ResultsSummary
from hdmi_channel_detector import HdmiChannelDetector, HdmiDeviceInfo

logger = logging.getLogger(__name__)


class SubwooferSwitchCallback(Protocol):
    """Called when the user needs to switch subwoofers before a measurement."""
    def __call__(self, target_sub_index: int, all_sub_indices: list[int]) -> None: ...


@dataclass
class ChannelConfig:
    """Configuration for a single speaker channel measurement."""
    channel_id: str                    # e.g. "fl", "c", "sw1"
    speaker_distance_m: float = 3.0    # estimated speaker-to-mic distance
    is_subwoofer: bool = False         # low-frequency channel — different processing
    reference_channel: bool = False    # use this as level/delay reference
    hdmi_channel_index: Optional[int] = None  # physical HDMI channel index (0-based)


@dataclass
class MeasurementResult:
    """Result of a single channel measurement."""
    channel_id: str
    ir: np.ndarray                     # impulse response (time domain)
    frequency_hz: np.ndarray            # frequency bins
    spl_db: np.ndarray                 # SPL curve (calibration applied if mic cal loaded)
    wav_path: Optional[Path] = None
    delay_ms: float = 0.0
    level_offset_db: float = 0.0
    distance_m: Optional[float] = None
    success: bool = False
    error_message: str = ""


@dataclass
class MeasurementConfig:
    """Global configuration for a measurement run."""
    channels: list[ChannelConfig] = field(default_factory=list)
    sweep_duration_sec: float = 6.0  # longer sweep for better low-freq resolution
    sweep_start_hz: float = 20.0
    sweep_end_hz: float = 20000.0
    sweep_amplitude_dbfs: float = -12.0
    sweep_amplitude_dbfs_lfe: float = -30.0  # much lower for LFE to avoid overdriving subwoofer
    suggested_avr_volume_db: float = -15.0   # master volume suggestion for safe SPL
    sample_rate: int = 48000
    buffer_size: int = 1024
    mic_calibration_path: Optional[str | Path] = None
    output_dir: str | Path = "./measurements"
    mic_positions: int = 1   # number of mic positions for spatial averaging
    subwoofer_switch_callback: Optional[SubwooferSwitchCallback] = None  # UI callback
    pause_for_subwoofer_switch: bool = True  # if True, pause and wait between subs
    num_subwoofers: int = 0  # number of subwoofers to measure (e.g. 2 or 4).
                              # If 0, auto-detected from HDMI. If HDMI reports fewer,
                              # only measure that many.
    use_sss: bool = False    # use Synchronized Swept Sine method (Novák 2015) for cleaner IR


class MeasurementOrchestrator:
    """High-level measurement controller.

    Usage:
        orch = MeasurementOrchestrator(config)
        orch.set_progress_callback(my_callback)

        # With user-guided subwoofer switching (recommended):
        orch.config.subwoofer_switch_callback = my_switch_callback
        orch.config.pause_for_subwoofer_switch = True

        results = orch.run()  # returns list of MeasurementResult
    """

    def __init__(self, config: MeasurementConfig):
        self.config = config
        self.engine = AudioEngine()
        self.processor = SignalProcessor(
            sample_rate=config.sample_rate,
            amplitude_dbfs=config.sweep_amplitude_dbfs,
            amplitude_dbfs_lfe=config.sweep_amplitude_dbfs_lfe,
            use_sss=config.use_sss,
        )
        self.calibration = MicCalibration()
        self.exporter = WavExporter(config.output_dir)
        self.results_summary = ResultsSummary()
        self._output_dir = Path(config.output_dir)
        self._progress_callback: Optional[Callable[[str, float], None]] = None
        self._cancelled = False
        self._subwoofer_indices: list[int] = []  # e.g. [1, 2] for 2-sub setup

        if config.mic_calibration_path:
            self.calibration.load_file(config.mic_calibration_path)

    def set_progress_callback(self, callback: Callable[[str, float], None]) -> None:
        self._progress_callback = callback

    def cancel(self) -> None:
        self._cancelled = True
        logger.info("Measurement cancelled")

    def _report_progress(self, message: str, fraction: float) -> None:
        if self._progress_callback:
            self._progress_callback(message, fraction)
        logger.info(f"[{fraction*100:.0f}%] {message}")

    # -------------------------------------------------------------------------
    # Device setup helpers
    # -------------------------------------------------------------------------

    def auto_configure_devices(
        self,
        playback_device_name_fragment: Optional[str] = None,
        capture_device_name_fragment: Optional[str] = None,
    ) -> bool:
        """Auto-detect and configure playback (AVR/HDMI) and capture (UMIK-1) devices."""
        # Find capture (UMIK-1)
        umik = self.engine.find_umik1()
        if umik:
            self.engine.set_capture_device(umik)
            logger.info(f"UMIK-1 detected: {umik.name}")
        elif capture_device_name_fragment:
            dev = self.engine.find_device_by_name(capture_device_name_fragment)
            if dev:
                self.engine.set_capture_device(dev)
        else:
            logger.warning("UMIK-1 not detected — set capture device manually")
            return False

        # Find playback (HDMI audio / AVR)
        if playback_device_name_fragment:
            dev = self.engine.find_device_by_name(playback_device_name_fragment)
            if dev:
                self.engine.set_playback_device(dev)
                logger.info(f"Playback device: {dev.name}")
        else:
            hdmi = self.engine.find_hdmi_audio()
            if hdmi:
                self.engine.set_playback_device(hdmi)
                logger.info(f"HDMI playback: {hdmi.name}")
            else:
                logger.warning("No HDMI/AVR playback device found — set manually")
                return False

        self.engine.set_sample_rate(self.config.sample_rate)
        self.engine.set_buffer_size(self.config.buffer_size)
        return True

    def detect_hdmi_channels(self, device_index: Optional[int] = None) -> HdmiDeviceInfo:
        """Detect all HDMI channels and subwoofer count for the playback device."""
        detector = HdmiChannelDetector()
        if device_index is not None:
            return detector.detect_channels(device_index)
        if self.engine.playback_device is not None:
            return detector.detect_channels(self.engine.playback_device.id)
        # Auto: first multichannel HDMI device
        devs = detector.list_hdmi_devices()
        if devs:
            return devs[0]
        mc = detector.list_all_multichannel_devices()
        if mc:
            return mc[0]
        raise RuntimeError("No HDMI/AVR multichannel device found")

    def build_channels_from_hdmi(
        self,
        device_info: HdmiDeviceInfo,
        include_subwoofers: bool = True,
    ) -> list[ChannelConfig]:
        """Build ChannelConfig list from HDMI device info.

        Args:
            device_info: HdmiDeviceInfo from detect_hdmi_channels()
            include_subwoofers: If True, include all detected subwoofer channels,
                               up to num_subwoofers from config.

        Returns:
            List of ChannelConfig, one per HDMI channel to measure.
        """
        configs = []
        sub_configs = []

        # Non-subwoofer channels first
        for ch in device_info.channels:
            if ch.is_subwoofer:
                continue
            configs.append(ChannelConfig(
                channel_id=ch.label.lower(),
                speaker_distance_m=3.0,
                is_subwoofer=False,
                hdmi_channel_index=ch.index,
            ))

        # Subwoofer channels — up to num_subwoofers from config
        num_requested = self.config.num_subwoofers
        for ch in device_info.channels:
            if not ch.is_subwoofer:
                continue
            if num_requested > 0 and len(sub_configs) >= num_requested:
                continue
            sub_configs.append(ChannelConfig(
                channel_id=ch.label.lower(),
                speaker_distance_m=3.0,
                is_subwoofer=True,
                hdmi_channel_index=ch.index,
            ))

        # Sort sub configs by sub_index so SW1, SW2, SW3, SW4 are in order
        def sub_sort_key(c: ChannelConfig) -> int:
            idx = self._get_sub_index_from_channel_id(c.channel_id)
            return idx if idx is not None else 0

        sub_configs.sort(key=sub_sort_key)
        configs.extend(sub_configs)

        # Track subwoofer indices for multi-sub switching prompts
        self._subwoofer_indices = [self._get_sub_index_from_channel_id(c.channel_id)
                                   for c in sub_configs
                                   if self._get_sub_index_from_channel_id(c.channel_id) is not None]
        self._subwoofer_indices = sorted(set(self._subwoofer_indices))

        logger.info(f"HDMI channels configured: {[c.channel_id for c in configs]}, "
                    f"subwoofers: {self._subwoofer_indices}")
        return configs

    # -------------------------------------------------------------------------
    # Per-channel measurement
    # -------------------------------------------------------------------------

    def measure_single_channel(
        self,
        channel: ChannelConfig,
        position_index: int = 0,
    ) -> MeasurementResult:
        """Measure a single channel: play sweep → capture → deconvolve → export."""
        result = MeasurementResult(
            channel_id=channel.channel_id,
            ir=np.array([]),
            frequency_hz=np.array([]),
            spl_db=np.array([]),
        )

        try:
            self._report_progress(f"Generating sweep for {channel.channel_id}...", 0.0)
            sweep = self.processor.generate_sweep(is_subwoofer=channel.is_subwoofer)

            self._report_progress(f"Playing sweep on {channel.channel_id}...", 0.1)

            num_frames = len(sweep)
            # sd.Lock serialises access to the same device across threads.
            # We lock the capture device (UMIK-1) so play+rec don't race,
            # and add a per-channel device flush between measurements so
            # PaErrorCode -9985 (Device unavailable) doesn't recur.
            import threading

            frames = int(self.config.sweep_duration_sec * self.config.sample_rate)
            playback_done = threading.Event()
            capture_done = threading.Event()
            capture_lock = threading.Lock()  # serialise capture device access

            def play_fn():
                try:
                    sd.play(sweep, device=self.engine.playback_device.id,
                           samplerate=self.config.sample_rate,
                           blocking=True)
                finally:
                    playback_done.set()

            def rec_fn():
                try:
                    with capture_lock:
                        recorded[:] = sd.rec(
                            frames=frames,
                            device=self.engine.capture_device.id,
                            samplerate=self.config.sample_rate,
                            channels=1,
                            dtype=np.float32,
                            blocking=True,
                        )
                finally:
                    capture_done.set()

            recorded = np.zeros((frames, 1), dtype=np.float32)
            rec_thread = threading.Thread(target=rec_fn, name=f"rec-{channel.channel_id}")
            play_thread = threading.Thread(target=play_fn, name=f"play-{channel.channel_id}")
            rec_thread.start()
            play_thread.start()
            play_thread.join()
            capture_done.wait()   # wait for rec to finish too
            rec_thread.join()

            # Flush both devices so next measurement starts clean.
            try:
                sd.flush(device=self.engine.playback_device.id)
            except Exception:
                pass
            try:
                sd.flush(device=self.engine.capture_device.id)
            except Exception:
                pass

            self._report_progress(f"Sweep complete, processing...", 0.6)

            ir = self.processor.measure_impulse_response(
                captured_mic=recorded,
                window_ms=50.0,
                fade_ms=5.0,
            )

            freq, spl = self.processor.get_frequency_response(ir)

            if self.calibration.is_loaded:
                spl = self.calibration.apply_from_arrays(freq, spl)

            dist_m = OffsetCalculator.estimate_distance_from_ir(ir, self.config.sample_rate)
            delay_ms = OffsetCalculator.calculate_delay_ms(
                distance_meters=dist_m,
                ref_distance_meters=0.0,
            )

            # Filename includes position index: FL0, FL1, SW1_0, SW1_1
            pos_label = str(position_index)
            fname_base = f"{channel.channel_id}_{pos_label}"
            wav_path = self.exporter.export_ir_wav(
                ir, fname_base, self.config.sample_rate, output_dir=self._output_dir
            )
            result.wav_path = wav_path
            result.distance_m = dist_m
            result.delay_ms = delay_ms
            result.ir = ir
            result.frequency_hz = freq
            result.spl_db = spl
            result.success = True
            self._report_progress(f"{channel.channel_id}: IR captured, delay={delay_ms:.1f}ms", 1.0)

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"Measurement failed for {channel.channel_id}: {e}")

        return result

    def _notify_subwoofer_switch(self, target_sub_index: int) -> None:
        """Show user instruction to switch subwoofers before measuring.

        Called before measuring each subwoofer when multiple are configured.
        The callback should display a message to the user and wait for confirmation
        before returning (so measurement proceeds only after the switch is done).

        If no callback is configured, logs the instruction instead.
        """
        all_subs = self._subwoofer_indices
        others = [s for s in all_subs if s != target_sub_index]
        msg = (f"SUBWOOFER SWITCH — Measure Subwoofer {target_sub_index} of {len(all_subs)}\n"
               f"\n"
               f"  ► Switch ON Subwoofer {target_sub_index}\n"
               f"  ► Switch OFF: {others if others else 'none'}\n"
               f"\n"
               f"Then press Enter / click Proceed to continue measuring...")

        if self.config.subwoofer_switch_callback:
            self.config.subwoofer_switch_callback(target_sub_index, all_subs)
        else:
            logger.info(msg)

    # -------------------------------------------------------------------------
    # Multi-channel / multi-position measurement
    # -------------------------------------------------------------------------

    def run(self) -> list[MeasurementResult]:
        """Run measurements for all channels at each position before moving the mic.

        Order: all channels at position 0 → all channels at position 1 → ...
        This keeps the mic in one place until all measurements at that spot are done.
        """
        all_results: list[MeasurementResult] = []
        sub_measured: set[int] = set()

        total_ops = len(self.config.channels) * self.config.mic_positions
        op_idx = 0

        for pos in range(self.config.mic_positions):
            for ch in self.config.channels:
                if self._cancelled:
                    break

                # ---- Pre-subwoofer measurement prompt (first time we see this sub) ----
                if ch.is_subwoofer and self.config.pause_for_subwoofer_switch:
                    sub_idx = self._get_sub_index_from_channel_id(ch.channel_id)
                    if sub_idx is not None and sub_idx not in sub_measured:
                        self._notify_subwoofer_switch(sub_idx)
                        sub_measured.add(sub_idx)

                op_idx += 1
                self._report_progress(
                    f"[Pos {pos+1}/{self.config.mic_positions}] Measuring {ch.channel_id}...",
                    op_idx / total_ops,
                )
                result = self.measure_single_channel(ch, position_index=pos)
                all_results.append(result)

            # Prompt user to move mic to next position
            if not self._cancelled and pos < self.config.mic_positions - 1:
                input(f"  ➤ Position {pos+1} complete — move mic to position {pos+2} and press ENTER...")

        return all_results

    def _get_sub_index_from_channel_id(self, channel_id: str) -> Optional[int]:
        """Extract subwoofer index from channel_id string (e.g. 'sw1' → 1, 'lfe' → 1)."""
        import re
        cid = channel_id.lower()
        if cid == "lfe":
            return 1
        m = re.search(r"sw(\d)", cid)
        if m:
            return int(m.group(1))
        return None

    def run_averaged(self) -> dict[str, MeasurementResult]:
        """Run measurements and return spatially-averaged result per channel."""
        raw_results = self.run()

        by_channel: dict[str, list[MeasurementResult]] = {}
        for r in raw_results:
            if r.success:
                by_channel.setdefault(r.channel_id, []).append(r)

        averaged: dict[str, MeasurementResult] = {}
        for ch_id, results in by_channel.items():
            if len(results) == 1:
                averaged[ch_id] = results[0]
                continue

            ir_stack = np.stack([r.ir for r in results if r.ir is not None], axis=0)
            avg_ir = np.mean(ir_stack, axis=0)

            freq = results[0].frequency_hz
            spl_stack = np.stack([r.spl_db for r in results if r.spl_db is not None], axis=0)
            avg_spl = np.mean(spl_stack, axis=0)

            avg_result = MeasurementResult(
                channel_id=ch_id,
                ir=avg_ir,
                frequency_hz=freq,
                spl_db=avg_spl,
                wav_path=results[0].wav_path,
                delay_ms=float(np.mean([r.delay_ms for r in results])),
                level_offset_db=float(np.mean([r.level_offset_db for r in results])),
                distance_m=float(np.mean([r.distance_m for r in results if r.distance_m])),
                success=True,
            )
            averaged[ch_id] = avg_result

        return averaged

    def run_selected_speakers(
        self,
        selected_channel_ids: list[str],
    ) -> list[MeasurementResult]:
        """Measure only specific speakers by name (e.g. ['FDL', 'FDR', 'SW1', 'SW2']).

        For subwoofers, triggers the subwoofer switching prompt before each unique sub.
        Useful for measuring a subset like height speakers or specific subwoofers.

        Args:
            selected_channel_ids: List of channel IDs to measure (e.g. ['FL','C','FR','SW1']).

        Returns:
            List of MeasurementResult, one per channel.
        """
        channels_to_measure = [
            ch for ch in self.config.channels
            if ch.channel_id.lower() in [c.lower() for c in selected_channel_ids]
        ]
        if not channels_to_measure:
            raise ValueError(f"None of {selected_channel_ids} found in config channels")

        # Temporarily replace config channels for this run
        original_channels = self.config.channels
        original_pause = self.config.pause_for_subwoofer_switch
        self.config.channels = channels_to_measure

        try:
            results = self.run()
        finally:
            self.config.channels = original_channels
            self.config.pause_for_subwoofer_switch = original_pause

        return results


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO)

    def demo_switch_callback(target: int, all_subs: list[int]) -> None:
        others = [s for s in all_subs if s != target]
        print(f"\n{'='*50}")
        print(f"🔌 SWITCH SUBWOOFERS BEFORE MEASURING")
        print(f"   ➤ Switch ON:  Subwoofer {target}")
        print(f"   ➤ Switch OFF: {others}")
        print(f"{'='*50}")
        input("   Press ENTER after switching subwoofers...")

    config = MeasurementConfig(
        sample_rate=48000,
        buffer_size=1024,
        pause_for_subwoofer_switch=True,
        suggested_avr_volume_db=-15.0,
        sweep_amplitude_dbfs=-12.0,
        sweep_amplitude_dbfs_lfe=-30.0,
    )
    config.subwoofer_switch_callback = demo_switch_callback

    orch = MeasurementOrchestrator(config)

    # ── Timestamped output folder ──────────────────────────────────────
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ts_folder = Path(config.output_dir) / ts
    ts_folder.mkdir(parents=True, exist_ok=True)
    orch._output_dir = ts_folder
    print(f"  Output folder: {orch._output_dir}")

    # ── Device selection ───────────────────────────────────────────────
    print("\n=== Device Configuration ===")

    # ── API preference ────────────────────────────────────────────────
    # WASAPI is the default because it properly exposes all HDMI/Atmos channels
    # on Windows. ASIO is also supported. If you have multichannel output via
    # a different API, use "any".
    api_opts = {
        "1": ("wasapi", "WASAPI (recommended — full Atmos support)"),
        "2": ("asio",   "ASIO (use if WASAPI doesn't show all channels)"),
        "3": ("any",    "Any API (auto-detect)"),
    }
    print("\n=== Audio API Preference ===")
    print("  WASAPI is recommended for HDMI/AVR receivers — it exposes all")
    print("  channel beds including Atmos height channels (5.1.2, 7.1.4, etc.)")
    for k, (api, desc) in api_opts.items():
        print(f"  [{k}] {desc}")
    api_choice = input(f"Audio API [1 for WASAPI, Enter for default]: ").strip() or "1"
    preferred_api = api_opts.get(api_choice, ("wasapi", ""))[0]
    print(f"  Using API: {preferred_api.upper()}")

    # ── Device selection (API-aware) ──────────────────────────────────
    detector = HdmiChannelDetector()
    print(f"\n=== Detecting {preferred_api.upper()} HDMI/AVR devices ===")

    # Show devices by preferred API first
    candidates: list[HdmiDeviceInfo] = []
    if preferred_api == "wasapi":
        candidates = (
            detector.list_devices_by_api("wasapi")
            or detector.list_devices_by_api("asio")
            or detector.list_all_multichannel_devices()
        )
    elif preferred_api == "asio":
        candidates = (
            detector.list_devices_by_api("asio")
            or detector.list_devices_by_api("wasapi")
            or detector.list_all_multichannel_devices()
        )
    else:
        candidates = detector.list_all_multichannel_devices()

    if not candidates:
        print("  No multichannel HDMI/AVR devices found!")
        print("  Check that:")
        print("    1. Your AVR/receiver is connected via HDMI")
        print("    2. HDMI audio is enabled in Windows Sound settings")
        print("    3. The correct HDMI input is selected on your AVR")
        raise RuntimeError("No HDMI/AVR multichannel device detected")

    print("\n  Available multichannel devices:")
    for i, d in enumerate(candidates):
        marker = " (recommended)" if i == 0 else ""
        print(f"  [{i}] {d.name} [{d.hostapi}] — {d.channel_count}ch — {d.layout_type}{marker}")

    sel_str = input(f"\n  Select device [0-{len(candidates)-1}, Enter for 0]: ").strip()
    sel_idx = int(sel_str) if sel_str else 0
    dev_info = candidates[sel_idx]

    # Configure engine with selected device
    pb_dev = AudioDevice(sd.query_devices(dev_info.device_index))
    orch.engine.set_playback_device(pb_dev)
    print(f"  Selected: {pb_dev.name} [{pb_dev.hostapi_name}] — {pb_dev.max_output_channels}ch")
    print(f"  Layout: {dev_info.layout_type}")

    # Capture device
    cap_dev = orch.engine.list_capture_devices()
    if not cap_dev:
        raise RuntimeError("No capture device found (is UMIK-1 connected?)")

    umik = next((d for d in cap_dev if "umik" in d.name.lower()), cap_dev[0])
    orch.engine.set_capture_device(umik)
    print(f"  Capture: {umik.name}")

    # ── Calibration file ───────────────────────────────────────────
    cal_defaults = [
        "calibrations/7150990_90deg.cal",
        os.path.join(os.path.expanduser("~"), "Downloads", "UMIK-1_calibration.txt"),
        "",
    ]
    cal_path = None
    for default in cal_defaults:
        if default and os.path.exists(default):
            cal_path = default
            break

    cal_prompt = input(f"\nCalibration file path [Enter for {cal_path or 'none'}]: ").strip()
    if cal_prompt:
        cal_path = cal_prompt

    if cal_path and os.path.exists(cal_path):
        # Detect UMIK-1 serial from cal filename for Full Scale SPL lookup
        import os as _os
        _serial = _os.path.basename(cal_path).split('_')[0] if cal_path else ''
        config.mic_calibration_path = cal_path
        orch.calibration.load_file(cal_path)
        sf = orch.calibration.sens_factor
        print(f"  Calibration loaded: {cal_path}")
        print(f"  Full Scale SPL: {sf:.2f} dB")
        print(f"  Cal offsets: {orch.calibration.frequency_range[0]:.0f}–"
              f"{orch.calibration.frequency_range[1]:.0f} Hz")
    else:
        print(f"  No calibration file — measurements will be uncorrected (relative dB)")
        print(f"  To fix SPL levels, load a .cal file and set the Sens Factor below")
        orch.calibration.is_loaded = False

    # ── SPL reference level (Sens Factor override) ──────────────────────
    # The cal file's Sens Factor handles absolute mic sensitivity.
    # If your AVR doesn't output exactly -12 dBFS at your reference volume,
    # or if your acoustic path differs from REW's, you can fine-tune here.
    # Set this to match REW's SPL reading or a calibrated SPL meter at 1 kHz.
    sf_prompt = input(
        f"\nFull Scale SPL override in dB (REW: 94 - dBFS @ 94 = Full Scale SPL)\n"
        f"  (Press Enter to keep {orch.calibration.sens_factor:.2f} dB, "
        f"or enter a value to fine-tune SPL levels): "
    ).strip()
    if sf_prompt:
        try:
            sf_override = float(sf_prompt)
            orch.calibration.reset_sens_factor(sf_override)
            print(f"  Sens Factor overridden to: {sf_override:.3f} dB re 1V/Pa")
        except ValueError:
            print(f"  Invalid value — keeping cal file Sens Factor")

    # ── AVR reference level reminder ───────────────────────────────────
    print(f"\n  ℹ  Set your AVR to your reference volume (e.g. -12 dBFS master volume)")
    print(f"     and use the same level for all measurements to get comparable SPL results.")

    # ── Build channel list from detected HDMI device info ───────────
    # Use HDMI channel mapping (not the generic layout_map) so all Atmos
    # height channels (FDL, FDR, SDL, SDR, etc.) are properly detected
    # based on the actual channel count reported by the WASAPI/ASIO device.
    dev_ch_list = dev_info.channels
    non_sub = [c for c in dev_ch_list if not c.is_subwoofer]
    sub_ch_list = [c for c in dev_ch_list if c.is_subwoofer]

    # Ask how many subwoofers to measure (if any detected)
    num_subwoofers = 0
    if sub_ch_list:
        print(f"\n  Subwoofer channels detected: {[c.label for c in sub_ch_list]}")
        num_sub_str = input(f"  Number of subwoofers to measure [1, max {len(sub_ch_list)}]: ").strip()
        num_subwoofers = int(num_sub_str) if num_sub_str else 1
        num_subwoofers = min(num_subwoofers, len(sub_ch_list))

    config.channels = [
        ChannelConfig(
            channel_id=c.label.lower(),
            is_subwoofer=c.is_subwoofer,
            speaker_distance_m=3.0,
        )
        for c in (non_sub + sub_ch_list[:num_subwoofers])
    ]

    # ── Subwoofer switching ─────────────────────────────────────────
    if num_subwoofers > 0:
        config.num_subwoofers = num_subwoofers
        config.pause_for_subwoofer_switch = True
    else:
        config.pause_for_subwoofer_switch = False

    # ── SSS mode prompt ──────────────────────────────────────────────
    sss_str = input("\nUse Synchronized Swept Sine method (SSS) for cleaner IR? [Y/n]: ").strip().lower()
    config.use_sss = sss_str != "n"
    if config.use_sss:
        print("  SSS mode enabled — distortion-corrected linear IR, best for Acourate FIR")
    else:
        print("  Standard exponential sweep mode selected")

    # ── Multi-position option ─────────────────────────────────────────
    num_positions_str = input(f"\nNumber of mic positions to measure [1]: ").strip()
    config.mic_positions = int(num_positions_str) if num_positions_str else 1
    if config.mic_positions < 1:
        config.mic_positions = 1

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\nOutput folder: {orch._output_dir}")
    print(f"\nChannels to measure:")
    for ch in config.channels:
        sub_tag = " (subwoofer)" if ch.is_subwoofer else ""
        print(f"  {ch.channel_id}{sub_tag}")
    print(f"  Positions per channel: {config.mic_positions}")
    if config.num_subwoofers > 0:
        print(f"  Subwoofer switching: ON (manual — watch for prompts)")

    print(f"\nSuggested AVR master volume: {config.suggested_avr_volume_db:.0f} dB")
    confirm = input("\nPress ENTER to start measurement... ").strip()

    # ── Measure ─────────────────────────────────────────────────────────
    results = orch.run()
    success = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\nMeasurement complete!")
    print(f"  {len(success)}/{len(results)} sweeps successful")
    if failed:
        print(f"  Failed channels: {[r.channel_id for r in failed]}")
    print(f"  Output folder: {orch._output_dir}")
    print(f"\n  WAV files ready for REW import:")
    for r in success:
        print(f"    {orch._output_dir.name}/{Path(r.wav_path).name}")
