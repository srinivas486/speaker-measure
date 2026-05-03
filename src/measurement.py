# Measurement Orchestrator — Phase 3
# speaker-measure
# Coordinates per-channel measurement: sweep → capture → deconvolve → export
# Handles multi-subwoofer measurement with user-guided subwoofer switching.

import numpy as np
import sounddevice as sd
from pathlib import Path
from dataclasses import dataclass, field
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
    sweep_duration_sec: float = 3.0
    sweep_start_hz: float = 20.0
    sweep_end_hz: float = 20000.0
    sweep_amplitude_dbfs: float = -12.0
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
        self.processor = SignalProcessor(sample_rate=config.sample_rate)
        self.calibration = MicCalibration()
        self.exporter = WavExporter(config.output_dir)
        self.results_summary = ResultsSummary()
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
            sweep = self.processor.generate_sweep()

            # Open stream for simultaneous play+rec
            stream = self.engine.open_stream(channels_playback=2, channels_capture=1)
            try:
                self._report_progress(f"Playing sweep on {channel.channel_id}...", 0.1)

                num_frames = len(sweep)
                recorded = sd.playrec(
                    sweep,
                    samplerate=self.config.sample_rate,
                    device=(
                        self.engine.playback_device.id,
                        self.engine.capture_device.id,
                    ),
                    dtype="float32",
                    blocking=True,
                )
                self._report_progress(f"Sweep complete, processing...", 0.6)

            finally:
                self.engine.close_stream()

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

            wav_path = self.exporter.export_ir_wav(ir, channel.channel_id, self.config.sample_rate)
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
        """Run measurements for all configured channels.

        For multi-subwoofer setups, prompts the user to switch subwoofers
        between each sub measurement so only the target sub plays.
        """
        all_results: list[MeasurementResult] = []
        sub_measured: set[int] = set()  # tracks which sub indices have been measured

        total_ops = len(self.config.channels) * self.config.mic_positions
        op_idx = 0

        for ch in self.config.channels:
            if self._cancelled:
                break

            # ---- Pre-subwoofer measurement prompt ----
            if ch.is_subwoofer and self.config.pause_for_subwoofer_switch:
                sub_idx = self._get_sub_index_from_channel_id(ch.channel_id)
                if sub_idx is not None and sub_idx not in sub_measured:
                    self._notify_subwoofer_switch(sub_idx)
                    sub_measured.add(sub_idx)

            for pos in range(self.config.mic_positions):
                if self._cancelled:
                    break

                op_idx += 1
                self._report_progress(
                    f"[Pos {pos+1}/{self.config.mic_positions}] Measuring {ch.channel_id}...",
                    op_idx / total_ops,
                )
                result = self.measure_single_channel(ch, position_index=pos)
                all_results.append(result)

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
        channels=[
            ChannelConfig(channel_id="fl", speaker_distance_m=3.2),
            ChannelConfig(channel_id="c", speaker_distance_m=3.0, reference_channel=True),
            ChannelConfig(channel_id="fr", speaker_distance_m=3.2),
            ChannelConfig(channel_id="sw1", is_subwoofer=True, speaker_distance_m=4.0),
            ChannelConfig(channel_id="sw2", is_subwoofer=True, speaker_distance_m=4.2),
        ],
        sample_rate=48000,
        buffer_size=1024,
        mic_positions=1,
        pause_for_subwoofer_switch=True,
    )
    config.subwoofer_switch_callback = demo_switch_callback

    orch = MeasurementOrchestrator(config)

    if orch.auto_configure_devices():
        print("Devices configured successfully")
        print(f"  Playback: {orch.engine.playback_device}")
        print(f"  Capture:  {orch.engine.capture_device}")
        results = orch.run()
        print(f"\nMeasurement complete! {len(results)} channels measured.")
    else:
        print("Device auto-config failed — check device connections")