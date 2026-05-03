# Mic Calibration — Phase 2
# speaker-measure
# Loads UMIK-1 calibration files and applies correction to frequency response

import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MicCalibration:
    """Loads and applies a microphone calibration response.

    Supports common formats:
    - REW .cal format (two-column: frequency_hz  sensitivity_offset_db)
    - UMIK-1 native CSV (has a header row, comma-separated)
    - Generic two-column whitespace-separated text files

    The calibration is stored as a lookup table: for each calibration frequency,
    store a dB offset to apply at that frequency. Interpolation is used between
    calibration points.
    """

    def __init__(self):
        self.cal_frequencies: Optional[np.ndarray] = None
        self.cal_offsets_db: Optional[np.ndarray] = None
        self.is_loaded: bool = False
        self._file_path: Optional[Path] = None

    def load_file(self, path: str | Path) -> bool:
        """Load calibration data from a file.

        Returns:
            True on success, False on failure.
        """
        path = Path(path)
        if not path.exists():
            logger.error(f"Calibration file not found: {path}")
            return False

        try:
            if path.suffix.lower() == ".cal":
                self._load_cal_format(path)
            elif path.suffix.lower() in (".csv", ".txt"):
                self._load_csv_or_txt(path)
            else:
                logger.error(f"Unknown calibration file format: {path.suffix}")
                return False

            self._file_path = path
            self.is_loaded = True
            logger.info(f"Loaded calibration from {path}: "
                        f"{len(self.cal_frequencies)} points, "
                        f"{self.cal_frequencies[0]:.1f} Hz → {self.cal_frequencies[-1]:.1f} Hz")
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration file {path}: {e}")
            return False

    def _load_cal_format(self, path: Path) -> None:
        """Parse REW .cal format: two columns (freq_hz, offset_db), no header."""
        data = np.loadtxt(path, ndmin=2)
        self.cal_frequencies = data[:, 0].astype(float)
        self.cal_offsets_db = data[:, 1].astype(float)
        self._sort()

    def _load_csv_or_txt(self, path: Path) -> None:
        """Parse generic CSV or two-column text file.

        Heuristic: if first non-empty line has text before the first comma,
        treat as a header and skip it.
        """
        with open(path, "r") as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        # Try to detect header
        data_lines = lines
        for i, line in enumerate(lines):
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    # If first two cols are numeric, this is data
                    if i > 0:
                        data_lines = lines[i:]
                    else:
                        data_lines = lines
                    break
                except ValueError:
                    continue

        data = []
        for line in data_lines:
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    data.append([float(parts[0]), float(parts[1])])
                except ValueError:
                    continue

        data = np.array(data)
        if len(data) == 0:
            raise ValueError("No valid frequency/offset pairs found")

        self.cal_frequencies = data[:, 0]
        self.cal_offsets_db = data[:, 1]
        self._sort()

    def _sort(self) -> None:
        idx = np.argsort(self.cal_frequencies)
        self.cal_frequencies = self.cal_frequencies[idx]
        self.cal_offsets_db = self.cal_offsets_db[idx]

    def get_offset(self, frequency_hz: float) -> float:
        """Get calibration offset (dB) at a given frequency using linear interpolation."""
        if not self.is_loaded:
            return 0.0
        # Linear interpolation in log-frequency space for smooth results
        log_f = np.log10(self.cal_frequencies)
        log_target = np.log10(frequency_hz)
        return float(np.interp(log_target, log_f, self.cal_offsets_db))

    def apply(self, frequency_hz: np.ndarray, spl_db: np.ndarray) -> np.ndarray:
        """Apply calibration offsets to an SPL curve.

        Args:
            frequency_hz: Array of frequency values (Hz)
            spl_db: Array of SPL values (dB) to correct

        Returns:
            Corrected SPL array.
        """
        if not self.is_loaded:
            return spl_db

        # Interpolate offsets for each frequency in the response
        offsets = np.array([self.get_offset(f) for f in frequency_hz])
        corrected = spl_db - offsets  # calibration file gives positive corrections (add to raw)
        logger.debug(f"Applied calibration: max offset {np.max(np.abs(offsets)):.2f} dB")
        return corrected

    def apply_from_arrays(
        self,
        frequency_hz: np.ndarray,
        spl_db: np.ndarray,
    ) -> np.ndarray:
        """Apply calibration from pre-loaded arrays (vectorised version)."""
        if not self.is_loaded:
            return spl_db

        log_f = np.log10(self.cal_frequencies)
        log_target = np.log10(np.maximum(frequency_hz, 1.0))  # guard against 0 Hz
        offsets = np.interp(log_target, log_f, self.cal_offsets_db)
        return spl_db - offsets

    @property
    def frequency_range(self) -> tuple[float, float]:
        """Return (min_freq_hz, max_freq_hz) of calibration data."""
        if not self.is_loaded:
            return 0.0, 0.0
        return float(self.cal_frequencies[0]), float(self.cal_frequencies[-1])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage — try loading a .cal file
    cal = MicCalibration()

    # Create a synthetic test calibration (flat ±0 for demo)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cal", delete=False, mode="w") as f:
        f.write("20 0.0\n100 0.0\n1000 0.0\n20000 0.0\n")
        cal_path = f.name

    cal.load_file(cal_path)
    print(f"Loaded: {cal.is_loaded}")
    print(f"Range: {cal.frequency_range}")
    print(f"Offset at 1kHz: {cal.get_offset(1000):.2f} dB")
    print(f"Offset at 100Hz: {cal.get_offset(100):.2f} dB")

    # Test apply
    import numpy as np
    freq = np.array([20, 100, 1000, 20000])
    spl = np.array([-20.0, -15.0, -10.0, -25.0])
    corrected = cal.apply(freq, spl)
    print(f"Corrected SPL: {corrected}")