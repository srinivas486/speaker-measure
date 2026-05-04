# Exporter — Phase 4
# speaker-measure
# Handles WAV export, delay/level offset calculation, and summary CSV generation

import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WavExporter:
    """Exports impulse responses and measurement results as REW-compatible WAV files.

    Files are written as:
        48 kHz, 24-bit PCM, mono (L=R identical if stereo is enabled)

    Naming convention:
        {channel_id}_{timestamp}.wav
        e.g. fl_20260501_133045.wav, sw1_20260501_133102.wav
    
    Stereo mode disabled by default — REW can import mono WAV files directly
    without incorrectly reading two identical channels as separate measurements.
    """

    EXPORT_SAMPLE_RATE = 48000
    EXPORT_BIT_DEPTH = 24  # 24-bit PCM

    def __init__(self, output_dir: str | Path = "./measurements"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def timestamp_filename(channel_id: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{channel_id.lower()}_{ts}"

    def export_ir_wav(
        self,
        ir: np.ndarray,
        channel_id: str,
        sample_rate: int = 48000,
        is_stereo: bool = False,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Export impulse response as 48 kHz mono WAV file.

        Args:
            ir: Impulse response, mono (N,) or stereo (N, 2)
            channel_id: Channel name with position index, e.g. 'FL0', 'SW1_1'
            sample_rate: Sample rate (default 48000)
            output_dir: Override output directory (default: use self.output_dir)

        Returns:
            Path to saved WAV file.
        """
        out_dir = output_dir if output_dir is not None else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{channel_id.lower()}.wav"
        out_path = out_dir / filename

        # Apply stereo if requested (REW accepts mono WAVs directly)
        if is_stereo:
            if ir.ndim == 1:
                ir = np.stack([ir, ir], axis=1)
            elif ir.ndim == 2 and ir.shape[1] == 1:
                ir = np.repeat(ir, 2, axis=1)

        sf.write(
            str(out_path),
            ir,
            sample_rate,
            subtype="PCM_24",
        )
        logger.info(f"Exported IR: {out_path} ({ir.shape[0]} samples, {ir.shape[1]} ch)")
        return out_path

    def export_frequency_data_csv(
        self,
        frequency_hz: np.ndarray,
        spl_db: np.ndarray,
        channel_id: str,
    ) -> Path:
        """Export frequency response as CSV (compatible with REW).

        Format: frequency_hz, spl_db
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{channel_id.lower()}_freq_{timestamp}.csv"
        out_path = self.output_dir / filename

        data = np.column_stack([frequency_hz, spl_db])
        header = "frequency_hz,spl_db"
        np.savetxt(str(out_path), data, header=header, comments="", delimiter=",", fmt="%.2f")
        logger.info(f"Exported frequency data: {out_path}")
        return out_path


class OffsetCalculator:
    """Calculates required speaker delay and level offsets relative to a reference channel.

    These are the values a user would input into their AVR or calibration software
    to align all speakers to the reference distance and level.
    """

    SPEED_OF_SOUND_MPS = 343.0  # at ~20°C indoor
    REFERENCE_CHANNEL = "fl"  # front left is default reference

    @staticmethod
    def calculate_delay_ms(distance_meters: float, ref_distance_meters: float = 0.0) -> float:
        """Calculate delay in milliseconds given speaker distance difference.

        Args:
            distance_meters: Speaker to mic distance (meters)
            ref_distance_meters: Reference speaker distance (meters)

        Returns:
            Delay in milliseconds (positive = speaker is further than reference)
        """
        delta_distance = distance_meters - ref_distance_meters
        delay_sec = delta_distance / OffsetCalculator.SPEED_OF_SOUND_MPS
        delay_ms = delay_sec * 1000.0
        return round(delay_ms, 2)

    @staticmethod
    def calculate_level_offset_db(
        measured_spl: float,
        ref_spl: float,
    ) -> float:
        """Calculate level offset in dB given measured SPL vs reference.

        Args:
            measured_spl: This speaker's measured SPL at reference frequency (e.g. 1 kHz)
            ref_spl: Reference channel SPL at same frequency

        Returns:
            Level offset in dB (positive = speaker is louder than reference)
        """
        return round(measured_spl - ref_spl, 2)

    @staticmethod
    def estimate_distance_from_ir(
        ir: np.ndarray,
        sample_rate: int,
        peak_threshold: float = 0.1,
    ) -> float:
        """Estimate speaker-to-mic distance from impulse response.

        The first large peak in the IR corresponds to direct sound arrival.
        We scan from the start and find the first sample exceeding the threshold.

        Args:
            ir: Impulse response (should be centred so peak is near t=0)
            sample_rate: Sample rate in Hz
            peak_threshold: Fraction of maximum peak to consider as "arrival"

        Returns:
            Estimated distance in metres.
        """
        if ir.ndim == 2:
            ir = ir[:, 0]  # use first channel

        ir_abs = np.abs(ir)
        peak = np.max(ir_abs)
        threshold = peak * peak_threshold

        # Find first sample above threshold
        arrival_idx = 0
        for i, val in enumerate(ir_abs):
            if val >= threshold:
                arrival_idx = i
                break

        time_sec = arrival_idx / sample_rate
        distance_m = time_sec * OffsetCalculator.SPEED_OF_SOUND_MPS
        return round(distance_m, 3)

    @staticmethod
    def calculate_all_delays(
        channel_distances: dict[str, float],
        ref_channel: str = "fl",
    ) -> dict[str, float]:
        """Calculate delays for all channels relative to reference.

        Args:
            channel_distances: Dict mapping channel_id -> distance_in_metres
            ref_channel: Reference channel (default "fl")

        Returns:
            Dict mapping channel_id -> delay_ms
        """
        ref_dist = channel_distances.get(ref_channel, 0.0)
        return {
            ch: OffsetCalculator.calculate_delay_ms(dist, ref_dist)
            for ch, dist in channel_distances.items()
        }


class ResultsSummary:
    """Collects and exports measurement results as CSV + summary."""

    def __init__(self):
        self.channels: dict[str, dict] = {}

    def add_channel(
        self,
        channel_id: str,
        wav_path: Path,
        delay_ms: float,
        level_offset_db: float,
        frequency_hz: Optional[np.ndarray] = None,
        spl_db: Optional[np.ndarray] = None,
        distance_m: Optional[float] = None,
    ) -> None:
        self.channels[channel_id] = {
            "wav_path": wav_path,
            "delay_ms": delay_ms,
            "level_offset_db": level_offset_db,
            "frequency_hz": frequency_hz,
            "spl_db": spl_db,
            "distance_m": distance_m,
        }

    def export_csv(self, output_path: Optional[str | Path] = None) -> Path:
        """Export summary as CSV file."""
        if output_path is None:
            output_path = Path("./measurements") / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        lines = ["channel,delay_ms,level_offset_db,distance_m,wav_file"]
        for ch_id, data in self.channels.items():
            lines.append(
                f"{ch_id},{data['delay_ms']},{data['level_offset_db']},"
                f"{data.get('distance_m', '')},{data['wav_path'].name}"
            )

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines))
        logger.info(f"Results CSV exported: {out_path}")
        return out_path

    def print_summary(self) -> None:
        print("\n=== Measurement Summary ===")
        print(f"{'Channel':<8} {'Delay (ms)':<12} {'Level (dB)':<12} {'Distance (m)':<12}")
        print("-" * 44)
        for ch_id, data in self.channels.items():
            dist = f"{data.get('distance_m', 0):.3f}" if data.get('distance_m') is not None else "—"
            print(f"{ch_id:<8} {data['delay_ms']:<12.2f} {data['level_offset_db']:<12.2f} {dist:<12}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    exporter = WavExporter("/tmp/speaker-measure-test")
    print(f"Output dir: {exporter.output_dir}")

    # Generate synthetic IR for testing
    sample_rate = 48000
    duration = 0.5  # 500 ms
    ir = np.random.randn(int(duration * sample_rate)).astype(np.float32) * 0.5

    path = exporter.export_ir_wav(ir, "fl", sample_rate)
    print(f"Saved: {path}")

    # Test offset calculator
    dists = {"fl": 3.5, "c": 3.2, "fr": 3.7, "sw1": 4.0}
    delays = OffsetCalculator.calculate_all_delays(dists)
    print("\nDelays:", delays)