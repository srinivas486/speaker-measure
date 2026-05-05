# Mic Calibration — Phase 3
# speaker-measure
# Loads UMIK-1 calibration files and applies absolute SPL correction

import re
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MicCalibration:
    """Loads and applies microphone calibration for absolute SPL calculation.

    Supports REW .cal files which contain:
      - Optional header line: "Sens Factor =X.XXXdB" — base mic sensitivity in dB re 1V/Pa
      - Data lines: frequency_hz  offset_db (per-frequency corrections)

    The Sens Factor (e.g. -1.135 dB) converts raw FFT magnitude (dBFS) to
    calibrated dB SPL. The per-frequency offsets then correct the UMIK-1's
    native response shape (positive = mic under-responds, add to compensate).

    Full SPL formula:
        SPL (dB) = raw_dBFS + Sens_Factor + cal_offsets(freq)

    Users set their AVR to a known reference level (e.g. -12 dBFS) so that
    measurements at the same volume setting are comparable and calibratable.
    """

    def __init__(self):
        self.cal_frequencies: Optional[np.ndarray] = None
        self.cal_offsets_db: Optional[np.ndarray] = None
        self.is_loaded: bool = False
        self._file_path: Optional[Path] = None
        self.sens_factor_db: float = 0.0  # base mic sensitivity from cal file header

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

            if self.cal_frequencies is None or len(self.cal_frequencies) == 0:
                logger.error(f"No calibration data loaded from {path}")
                return False

            self._file_path = path
            self.is_loaded = True
            logger.info(f"Loaded calibration from {path}: "
                        f"{len(self.cal_frequencies)} points, "
                        f"{self.cal_frequencies[0]:.1f} Hz → {self.cal_frequencies[-1]:.1f} Hz, "
                        f"Sens Factor = {self.sens_factor_db:.3f} dB re 1V/Pa")
            return True

        except Exception as e:
            logger.error(f"Failed to load calibration file {path}: {e}")
            return False

    def _load_cal_format(self, path: Path) -> None:
        """Parse REW .cal format: optional header, then two columns (freq_hz, offset_db).

        The header (if present) contains the Sens Factor, e.g.:
            "Sens Factor =-1.135dB, SERNO: 7150990"
        """
        data = []
        self.sens_factor_db = 0.0  # default: no sensitivity adjustment

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Check for Sens Factor header line
                if 'Sens Factor' in line or 'sens' in line.lower():
                    m = re.search(r'[+-]?\d+\.?\d*', line.split('Sens Factor')[-1])
                    if m:
                        self.sens_factor_db = float(m.group())
                        logger.info(f"Sens Factor from cal file: {self.sens_factor_db:.3f} dB re 1V/Pa")
                    continue
                # Skip comment lines
                if line.startswith('#'):
                    continue
                parts = line.replace(',', ' ').split()
                if len(parts) >= 2:
                    try:
                        data.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        continue

        if not data:
            raise ValueError(f"No calibration data found in {path}")

        data = np.array(data)
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

        data_lines = lines
        for i, line in enumerate(lines):
            parts = line.replace(",", " ").split()
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    float(parts[1])
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
        log_f = np.log10(self.cal_frequencies)
        log_target = np.log10(frequency_hz)
        return float(np.interp(log_target, log_f, self.cal_offsets_db))

    def apply(self, frequency_hz: np.ndarray, spl_db: np.ndarray) -> np.ndarray:
        """Apply calibration: convert raw dBFS to calibrated dB SPL.

        Full formula: SPL = raw_dBFS + Sens_Factor + cal_offsets

        The Sens Factor (from cal file header, e.g. -1.135 dB) handles absolute
        SPL conversion. The per-frequency offsets correct the mic's native
        frequency response shape.

        Args:
            frequency_hz: Array of frequency values (Hz)
            spl_db: Raw SPL values in dBFS (from FFT magnitude)

        Returns:
            Calibrated SPL array in dB.
        """
        if not self.is_loaded:
            return spl_db

        offsets = np.array([self.get_offset(f) for f in frequency_hz])
        corrected = spl_db + self.sens_factor_db + offsets
        logger.debug(f"Calibration applied: Sens Factor={self.sens_factor_db:.3f} dB, "
                     f"max offset={np.max(np.abs(offsets)):.2f} dB")
        return corrected

    def apply_from_arrays(
        self,
        frequency_hz: np.ndarray,
        spl_db: np.ndarray,
    ) -> np.ndarray:
        """Vectorised version: SPL = raw_dBFS + Sens_Factor + cal_offsets(freq)."""
        if not self.is_loaded:
            return spl_db

        log_f = np.log10(np.maximum(frequency_hz, 1.0))
        log_c = np.log10(self.cal_frequencies)
        offsets = np.interp(log_f, log_c, self.cal_offsets_db)
        return spl_db + self.sens_factor_db + offsets

    @property
    def sens_factor(self) -> float:
        """Return the base mic sensitivity in dB re 1V/Pa (from cal file header)."""
        return self.sens_factor_db

    @property
    def frequency_range(self) -> tuple[float, float]:
        """Return (min_freq_hz, max_freq_hz) of calibration data."""
        if not self.is_loaded:
            return 0.0, 0.0
        return float(self.cal_frequencies[0]), float(self.cal_frequencies[-1])

    def reset_sens_factor(self, value_db: float) -> None:
        """Override the Sens Factor with a custom value (e.g. from SPL meter calibration).

        Call this after load_file() to override the cal file's Sens Factor with a
        value measured using a reference SPL meter at a known volume setting.
        """
        self.sens_factor_db = value_db
        logger.info(f"Sens Factor overridden to {value_db:.3f} dB re 1V/Pa")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cal = MicCalibration()

    # Create synthetic test calibration
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".cal", delete=False, mode="w") as f:
        f.write('"Sens Factor =-1.135dB, SERNO: 7150990"\n')
        f.write("20 0.1\n100 0.5\n1000 0.0\n5000 0.8\n10000 0.0\n20000 -4.3\n")
        cal_path = f.name

    cal.load_file(cal_path)
    print(f"Loaded: {cal.is_loaded}")
    print(f"Range: {cal.frequency_range}")
    print(f"Sens Factor: {cal.sens_factor:.3f} dB re 1V/Pa")
    print(f"Offset at 1kHz: {cal.get_offset(1000):.2f} dB")
    print(f"Offset at 20kHz: {cal.get_offset(20000):.2f} dB")

    # Test apply
    freq = np.array([20, 100, 1000, 5000, 10000, 20000])
    raw_spl = np.array([-20.0, -15.0, -10.0, -8.0, -6.0, -25.0])
    corrected = cal.apply(freq, raw_spl)
    expected = raw_spl + (-1.135) + np.array([0.1, 0.5, 0.0, 0.8, 0.0, -4.3])
    print(f"Raw SPL:     {raw_spl}")
    print(f"Corrected:   {corrected}")
    print(f"Expected:    {expected}")
    print(f"Match: {np.allclose(corrected, expected)}")