# Signal Processor — Phase 2
# speaker-measure
# Generates measurement sweeps, performs deconvolution, windowing, and frequency response extraction

import numpy as np
from scipy import signal as sp_signal
from scipy.io import wavfile
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SweepGenerator:
    """Generates exponential sine sweeps for room measurement.

    Exponential sweeps provide:
    - Flat frequency coverage (20 Hz to 20 kHz in a single sweep)
    - High SNR (energy concentrated at low frequencies where room effects are strongest)
    - Easy deconvolution via inverse filter
    """

    DEFAULT_DURATION_SEC = 3.0
    DEFAULT_START_HZ = 20.0
    DEFAULT_END_HZ = 20000.0
    DEFAULT_AMPLITUDE_DBFS = -12.0  # Peak at -12 dBFS to avoid clipping

    def __init__(
        self,
        sample_rate: int = 48000,
        duration_sec: float = DEFAULT_DURATION_SEC,
        start_hz: float = DEFAULT_START_HZ,
        end_hz: float = DEFAULT_END_HZ,
        amplitude_dbfs: float = DEFAULT_AMPLITUDE_DBFS,
    ):
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.start_hz = start_hz
        self.end_hz = end_hz
        self.amplitude_dbfs = amplitude_dbfs

    @property
    def amplitude_linear(self) -> float:
        """Peak amplitude as linear scale (1.0 = 0 dBFS)."""
        return 10 ** (self.amplitude_dbfs / 20.0)

    @property
    def num_samples(self) -> int:
        return int(self.duration_sec * self.sample_rate)

    @property
    def sweep_factor(self) -> float:
        """The sweep's exponent factor: f(t) = start * exp(t * K), K = ln(end/start) / T"""
        return np.log(self.end_hz / self.start_hz) / self.duration_sec

    def generate(self) -> np.ndarray:
        """Generate a single exponential sine sweep.

        Mathematical form:
            f(t) = start_hz * exp(t * K)          [instantaneous frequency]
            θ(t) = ∫ f(τ) dτ = (start_hz / K) * (exp(K*t) - 1)  [phase]
            sweep(t) = amplitude * sin(2π * θ(t))

        Returns:
            Stereo array of shape (num_samples, 2), amplitude-normalised.
        """
        t = np.arange(self.num_samples) / self.sample_rate
        K = self.sweep_factor
        # Phase: integral of f(t) = start_hz/K * (exp(K*t) - 1)
        phase = (self.start_hz / K) * (np.exp(K * t) - 1.0)
        sweep = np.sin(2.0 * np.pi * phase)
        sweep *= self.amplitude_linear
        # Make stereo (duplicate on both channels for HDMI output)
        stereo = np.stack([sweep, sweep], axis=1)
        logger.info(f"Generated sweep: {self.num_samples} samples ({self.duration_sec}s), "
                    f"{self.start_hz} Hz → {self.end_hz} Hz, amplitude {self.amplitude_dbfs} dBFS")
        return stereo.astype(np.float32)

    def get_inverse_filter(self) -> np.ndarray:
        """Generate the inverse filter for deconvolution.

        The inverse filter is the time-reversed, amplitude-normalised sweep.
        In frequency domain: H(f) = 1 / S(f), where S(f) is sweep spectrum.
        Time-domain version (windowed) is used for cleaner IR results.

        Returns:
            Array of shape (num_samples, 2) — time-reversed sweep.
        """
        sweep = self.generate()
        # Time-reverse and normalise to prevent energy blow-up in deconvolution
        inv = np.flipud(sweep) / (np.sum(sweep[:, 0] ** 2) + 1e-12)
        logger.info(f"Inverse filter generated, length: {len(inv)}")
        return inv.astype(np.float32)

    def save_wav(self, path: str | Path, data: Optional[np.ndarray] = None) -> None:
        """Save sweep to WAV file."""
        if data is None:
            data = self.generate()
        wavfile.write(str(path), self.sample_rate, data)
        logger.info(f"Sweep saved to {path}")


class Deconvolver:
    """Performs frequency-domain deconvolution to extract impulse response from captured measurement."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def deconvolve(
        self,
        captured: np.ndarray,
        sweep: np.ndarray,
        window_ms: float = 50.0,
        fade_ms: float = 5.0,
    ) -> np.ndarray:
        """Deconvolve impulse response from captured mic signal.

        Args:
            captured: Recorded microphone signal, shape (N,) or (N, 1)
            sweep: The sweep signal that was played (same length as captured ideally)
            window_ms: Time-gating window in milliseconds — keep IR up to this point,
                       discard reflections that arrive after this
            fade_ms: Cosine fade-out duration at end of window

        Returns:
            Impulse response array (time-domain).
        """
        if captured.ndim == 2:
            captured = captured[:, 0]  # use first channel (mono)

        # ---- Frequency domain deconvolution ----
        # Use the original sweep as reference for cleanest results
        if len(sweep) == 0:
            raise ValueError("Empty sweep")

        # Zero-pad to same length
        N = max(len(captured), len(sweep))
        cap_pad = np.pad(captured, (0, N - len(captured)))
        swp_pad = np.pad(sweep[:, 0] if sweep.ndim == 2 else sweep, (0, N - len(sweep)))

        # FFT of captured and sweep
        Cap = np.fft.rfft(cap_pad)
        Swp = np.fft.rfft(swp_pad)

        # Inverse filter in frequency domain (with safety floor to avoid /0)
        floor = 1e-12 * np.max(np.abs(Swp))
        inv_spectrum = np.conj(Swp) / (np.abs(Swp) ** 2 + floor)

        # Multiply captured by inverse filter → IR in frequency domain
        IR = Cap * inv_spectrum

        # Time-domain IR
        ir_time = np.fft.irfft(IR, n=N)
        ir_time = np.roll(ir_time, len(sweep) // 2)  # centre the IR

        # ---- Time-gating: window to remove reflections ----
        window_samples = int(window_ms * self.sample_rate / 1000)
        fade_samples = int(fade_ms * self.sample_rate / 1000)
        ir_time[window_samples:] = 0

        # Apply cosine fade at end of window
        if fade_samples > 0 and window_samples > fade_samples:
            fade_win = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_samples) / fade_samples))
            ir_time[window_samples - fade_samples:window_samples] *= fade_win

        logger.info(f"Deconvolution complete: IR length={len(ir_time)}, window={window_ms}ms")
        return ir_time.astype(np.float32)

    def extract_frequency_response(
        self,
        ir: np.ndarray,
        smoothing_octaves: float = 1.0 / 3.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert impulse response to frequency response (SPL-style magnitude).

        Args:
            ir: Time-domain impulse response from deconvolve()
            smoothing_octaves: Fractional-octave smoothing (1/3 = standard 1/3-octave display)

        Returns:
            Tuple of (frequencies_hz, spl_db).
        """
        # FFT of IR → magnitude spectrum
        spectrum = np.fft.rfft(ir)
        magnitude = np.abs(spectrum)
        freq_bins = np.fft.rfftfreq(len(ir), d=1.0 / self.sample_rate)

        # Convert to dB (relative, not absolute SPL — no calibration applied here)
        spl = 20.0 * np.log10(magnitude + 1e-12)

        # ---- Smooth in log-frequency space ----
        if smoothing_octaves > 0:
            spl = self._smooth_octave(freq_bins, spl, smoothing_octaves)

        return freq_bins, spl

    def _smooth_octave(self, freq: np.ndarray, spl: np.ndarray, octaves: float) -> np.ndarray:
        """Apply fractional-octave smoothing in log-frequency domain."""
        # Build 1/3-octave bands
        log_f = np.log10(freq + 1)
        log_min, log_max = log_f[0], log_f[-1]

        # Number of bands from 20 Hz to 20 kHz at given octave fraction
        # band_width = octaves * log10(2)
        band_width = octaves * 0.3010  # log10(2) ≈ 0.3010
        n_bands = int(round((log_max - log_min) / band_width))

        smoothed = np.zeros_like(spl)
        for i in range(len(freq)):
            # Find frequencies within ±band_width/2 of this freq
            log_center = log_f[i]
            log_low = log_center - band_width / 2
            log_high = log_center + band_width / 2
            mask = (log_f >= log_low) & (log_f <= log_high)
            if np.any(mask):
                smoothed[i] = np.mean(spl[mask])
            else:
                smoothed[i] = spl[i]

        return smoothed


class SignalProcessor:
    """High-level signal processor coordinating sweep generation and deconvolution."""

    def __init__(
        self,
        sample_rate: int = 48000,
        amplitude_dbfs: float = -12.0,
        amplitude_dbfs_lfe: float = -30.0,
    ):
        self.sample_rate = sample_rate
        self.amplitude_dbfs = amplitude_dbfs
        self.amplitude_dbfs_lfe = amplitude_dbfs_lfe
        self.sweep_gen = SweepGenerator(
            sample_rate=sample_rate,
            amplitude_dbfs=amplitude_dbfs,
        )
        self.deconvolver = Deconvolver(sample_rate=sample_rate)

    def generate_sweep(self, is_subwoofer: bool = False) -> np.ndarray:
        """Generate sweep with amplitude appropriate for the channel type.

        Subwoofers get a much lower amplitude sweep to avoid overdriving the driver.
        """
        amp = self.amplitude_dbfs_lfe if is_subwoofer else self.amplitude_dbfs
        # Only recreate SweepGenerator if amplitude changed
        if not hasattr(self, '_sweep_gen') or self._sweep_gen.amplitude_dbfs != amp:
            self._sweep_gen = SweepGenerator(sample_rate=self.sample_rate, amplitude_dbfs=amp)
            self._inv_filter = None  # reset cached inverse filter
        return self._sweep_gen.generate()

    def get_inverse_filter(self) -> np.ndarray:
        """Get the inverse filter for the current sweep (cached)."""
        if not hasattr(self, '_inv_filter') or self._inv_filter is None:
            self._inv_filter = self._sweep_gen.get_inverse_filter()
        return self._inv_filter

    def generate_sweep(self) -> np.ndarray:
        return self.sweep_gen.generate()

    def measure_impulse_response(
        self,
        captured_mic: np.ndarray,
        window_ms: float = 50.0,
        fade_ms: float = 5.0,
    ) -> np.ndarray:
        inv = self.get_inverse_filter()
        return self.deconvolver.deconvolve(captured_mic, inv, window_ms, fade_ms)

    def get_frequency_response(self, ir: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.deconvolver.extract_frequency_response(ir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    proc = SignalProcessor()

    print("Generating sweep...")
    sweep = proc.generate_sweep()
    print(f"Sweep shape: {sweep.shape}, peak: {np.max(np.abs(sweep)):.4f}")

    print("Generating inverse filter...")
    inv = proc.sweep_gen.get_inverse_filter()
    print(f"Inverse filter shape: {inv.shape}")

    proc.sweep_gen.save_wav("/tmp/test_sweep.wav")
    print("Saved test_sweep.wav")

    # Quick test: synthetic "captured" signal = sweep (for testing pipeline)
    print("Simulating deconvolution with clean sweep...")
    captured = sweep[:, 0]
    ir = proc.deconvolver.deconvolve(captured, sweep)
    freq, spl = proc.get_frequency_response(ir)
    print(f"Frequency range: {freq[0]:.1f} Hz → {freq[-1]:.1f} Hz")
    print(f"SPL range: {np.min(spl):.1f} → {np.max(spl):.1f} dB")