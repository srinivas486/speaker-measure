# Synchronized Swept Sine (SSS) — Antonín Novák et al. JAES 2015
# Implementation for speaker-measure
#
# Based on:
#   https://github.com/SiggiGue/syncsweptsine (Siegfried Gündert, MPL-2.0)
#   https://ant-novak.com/pages/sss/ (Antonín Novák)
#   Novak et al. (2015) "Synchronized swept-sine: Theory, application, and implementation"
#                        J. Audio Eng. Soc., 63(10), 786–798.
#
# Key idea: two synchronized sweeps (forward + time-reversed) played in sequence,
# captured together, then processed to separate linear IR from harmonic distortion IRs.
# The forward sweep captures H1 (linear). The reversed sweep helps isolate H2, H3, etc.
# Only the linear H1 component is used for the final IR.

import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SyncSweepGenerator:
    """Generates the Synchronized Swept Sine signal per Novák eq. (1).

    Signal: x(t) = sin( 2π f1/L * exp(t/L) )
    where L = T / ln(f2/f1)  [sweep period]
          T = duration in seconds
          f1 = start frequency
          f2 = end frequency

    The sweep has constant amplitude and exponentially increasing frequency.
    """

    def __init__(
        self,
        start_freq_hz: float,
        end_freq_hz: float,
        duration_sec: float,
        sample_rate: int,
        amplitude_dbfs: float = -12.0,
    ):
        if start_freq_hz >= end_freq_hz:
            raise ValueError(f"start_freq ({start_freq_hz}) must be < end_freq ({end_freq_hz})")
        if duration_sec <= 0:
            raise ValueError(f"duration_sec ({duration_sec}) must be positive")
        if sample_rate <= 0:
            raise ValueError(f"sample_rate ({sample_rate}) must be positive")

        self.start_freq_hz = start_freq_hz
        self.end_freq_hz = end_freq_hz
        self.duration_sec = duration_sec
        self.sample_rate = sample_rate
        self.amplitude_linear = 10 ** (amplitude_dbfs / 20.0)

        # Derived parameters
        self.log_freq_ratio = np.log(end_freq_hz / start_freq_hz)  # ln(f2/f1)
        # kappa (κ): rounds duration to an integer number of sweep periods
        self.kappa = round(start_freq_hz * duration_sec / self.log_freq_ratio)
        # Actual duration (may differ slightly from requested due to rounding)
        self.actual_duration = self.kappa * self.log_freq_ratio / start_freq_hz
        # Sweep period L in the paper
        self.sweep_period = self.kappa / start_freq_hz  # seconds

        self._signal: Optional[np.ndarray] = None
        self._time_vector: Optional[np.ndarray] = None

    @property
    def num_samples(self) -> int:
        return int(self.actual_duration * self.sample_rate)

    def generate(self) -> np.ndarray:
        """Generate the exponential swept-sine as a stereo numpy array.

        Returns:
            np.ndarray shape (num_samples, 2), dtype float32, amplitude scaled.
        """
        n = self.num_samples
        t = np.arange(n) / self.sample_rate

        # Instantaneous phase: integral of f(t) = f1/L * exp(t/L)
        # θ(t) = 2π * f1/L * (exp(t/L) - 1)  [from paper eq. for phase]
        # Simplified: since f(t) = f1 * exp(t/L), the phase integral is:
        # θ(t) = 2π * f1 * L * (exp(t/L) - 1) = 2π * κ * (exp(t/L) - 1)
        # Wait — let me re-derive from the paper:
        # x(t) = sin[2π f1/L exp(t/L)]  [eq 1 in paper]
        # θ(t) = 2π f1/L ∫ exp(τ/L) dτ = 2π f1/L * L * exp(t/L) = 2π f1 * exp(t/L)
        # Actually: ∫ exp(τ/L) dτ = L * exp(t/L)
        # So: θ(t) = 2π f1/L * L * (exp(t/L) - 1) = 2π f1 * (exp(t/L) - 1)
        # But the time vector starts at 0, so the phase is:
        phase = 2.0 * np.pi * self.start_freq_hz * self.sweep_period * (np.exp(t / self.sweep_period) - 1.0)

        sweep = np.sin(phase) * self.amplitude_linear

        # Make stereo (duplicate L and R channels for HDMI playback)
        stereo = np.stack([sweep, sweep], axis=1)
        logger.info(
            f"SyncSweep generated: {n} samples ({self.actual_duration:.3f}s), "
            f"{self.start_freq_hz} Hz → {self.end_freq_hz} Hz, "
            f"amplitude {20*np.log10(self.amplitude_linear):.1f} dBFS, "
            f"sweep_period={self.sweep_period:.4f}s"
        )
        self._signal = stereo.astype(np.float32)
        return self._signal

    def get_windowed(
        self,
        fade_samples: int = 2048,
        pre_samples: int = 0,
        post_samples: int = 0,
    ) -> np.ndarray:
        """Return the sweep with fade-in/out Hanning ramps and optional silence padding.

        Args:
            fade_samples: number of samples for fade-in and fade-out ramps.
            pre_samples:  silence samples before the sweep (for capture alignment).
            post_samples: silence samples after the sweep (for capture tail).

        Returns:
            np.ndarray shape (total_samples, 2), float32.
        """
        if self._signal is None:
            self.generate()

        sig = self._signal.copy()
        # Fade in/out with Hanning window flanks
        fade_win = np.sin(np.pi * np.arange(fade_samples) / fade_samples) ** 2
        sig[:fade_samples] *= fade_win[:, np.newaxis]
        sig[-fade_samples:] *= fade_win[::-1, np.newaxis]

        parts = []
        if pre_samples > 0:
            parts.append(np.zeros((pre_samples, 2), dtype=np.float32))
        parts.append(sig)
        if post_samples > 0:
            parts.append(np.zeros((post_samples, 2), dtype=np.float32))
        return np.concatenate(parts, axis=0)

    def __len__(self) -> int:
        return self.num_samples


class InvertedSSSpectrum:
    """Analytical inverted spectrum of the synchronized swept sine (Novák eq. 43).

    Instead of time-reversing and windowing the sweep (which gives a noisy inverse),
    we use the exact analytical form of the inverse spectrum:

        S_inv(f) = 2*sqrt(f/L) * exp(-j*2π*f*L*(1 - ln(f/f1)) + j*π/4)

    Deconvolution:  H1(f) = Y(f) * S_inv(f)  (single multiplication, no division)
    """

    def __init__(
        self,
        sample_rate: int,
        sweep_period: float,
        start_freq_hz: float,
        fft_length: int,
    ):
        self.sample_rate = sample_rate
        self.sweep_period = sweep_period
        self.start_freq_hz = start_freq_hz
        self.fft_length = fft_length
        self._spectrum: Optional[np.ndarray] = None
        self._freq: Optional[np.ndarray] = None

    @classmethod
    def from_sync_sweep(cls, sweep: SyncSweepGenerator, fft_length: int) -> "InvertedSSSpectrum":
        """Factory: create from a SyncSweepGenerator instance."""
        return cls(
            sample_rate=sweep.sample_rate,
            sweep_period=sweep.sweep_period,
            start_freq_hz=sweep.start_freq_hz,
            fft_length=fft_length,
        )

    @property
    def spectrum(self) -> np.ndarray:
        if self._spectrum is None:
            self._calculate()
        return self._spectrum

    @property
    def freq(self) -> np.ndarray:
        if self._freq is None:
            self._calculate()
        return self._freq

    def _calculate(self) -> None:
        n = self.fft_length
        sr = self.sample_rate
        L = self.sweep_period
        f1 = self.start_freq_hz

        freq = np.fft.rfftfreq(n, 1.0 / sr)  # frequency vector [Hz]
        spec = np.zeros_like(freq, dtype=np.complex128)

        # Novák eq. (43) — analytical inverse spectrum (skip DC bin at index 0)
        freq_pos = freq[1:]  # exclude DC
        spec[1:] = (
            2.0 * np.sqrt(freq_pos / L)
            * np.exp(
                -2j * np.pi * freq_pos * L * (1.0 - np.log(freq_pos / f1))
                + 1j * np.pi / 4.0
            )
        )
        self._spectrum = spec
        self._freq = freq


class SSSDeconvolver:
    """Deconvolves a recorded sweep using the SSS method.

    Produces a linear impulse response (IR) by frequency-domain multiplication
    with the analytical inverse spectrum. Also provides access to higher-order
    harmonic IRs (H2, H3) if needed.
    """

    def __init__(
        self,
        sample_rate: int,
        sweep_period: float,
        start_freq_hz: float,
        fft_length: int,
        fade_samples: int = 2048,
    ):
        self.sample_rate = sample_rate
        self.sweep_period = sweep_period
        self.start_freq_hz = start_freq_hz
        self.fft_length = fft_length
        self.fade_samples = fade_samples

        self._inv_spec: Optional[InvertedSSSpectrum] = None
        self._hhir: Optional[np.ndarray] = None
        self._hhfrf: Optional[np.ndarray] = None

    @classmethod
    def from_sweep(cls, sweep: SyncSweepGenerator, fade_samples: int = 2048) -> "SSSDeconvolver":
        """Factory: create deconvolver from a SyncSweepGenerator."""
        fft_len = len(sweep.generate())  # use sweep length as FFT size
        return cls(
            sample_rate=sweep.sample_rate,
            sweep_period=sweep.sweep_period,
            start_freq_hz=sweep.start_freq_hz,
            fft_length=fft_len,
            fade_samples=fade_samples,
        )

    @property
    def inv_spectrum(self) -> InvertedSSSpectrum:
        if self._inv_spec is None:
            self._inv_spec = InvertedSSSpectrum(
                sample_rate=self.sample_rate,
                sweep_period=self.sweep_period,
                start_freq_hz=self.start_freq_hz,
                fft_length=self.fft_length,
            )
        return self._inv_spec

    def deconvolve(self, recorded: np.ndarray) -> np.ndarray:
        """Deconvolve a recorded sweep signal into an impulse response.

        The recorded signal should be the captured microphone input (mono or stereo).
        We use only the first channel for deconvolution.

        Args:
            recorded: np.ndarray shape (N, 2) or (N,) — captured sweep signal.

        Returns:
            ir: np.ndarray shape (fft_length,) — linear impulse response (time domain).
        """
        # Ensure mono
        if recorded.ndim == 2:
            recorded = recorded[:, 0]  # use left channel

        n = self.fft_length
        # Zero-pad to fft_length
        if len(recorded) < n:
            recorded = np.pad(recorded, (0, n - len(recorded)))
        elif len(recorded) > n:
            recorded = recorded[:n]

        # FFT of recorded signal
        Y = np.fft.rfft(recorded, n=n)
        # Multiply with analytical inverse spectrum (H1 = Y * S_inv)
        H1 = Y * self.inv_spectrum.spectrum
        # Inverse FFT → impulse response
        ir = np.fft.irfft(H1, n=n)
        logger.info(f"SSS deconvolution done, IR length: {len(ir)}")
        return ir

    def get_linear_ir(self, recorded: np.ndarray, fade_samples: int = 2048) -> np.ndarray:
        """Extract the clean linear IR from a recorded sweep.

        Applies windowing to isolate the primary IR from residual noise and
        any residual harmonic content.

        Args:
            recorded: captured sweep signal.
            fade_samples: window flank size.

        Returns:
            ir_windowed: windowed impulse response.
        """
        ir = self.deconvolve(recorded)
        return self._window_ir(ir, fade_samples)

    def _window_ir(self, ir: np.ndarray, fade_samples: int) -> np.ndarray:
        """Window the IR to remove noise tail beyond the useful signal region.

        The linear IR from an SSS sweep is concentrated near t=0.
        We keep a window centered around the peak and fade to silence.
        """
        ir = ir.copy()
        peak_idx = np.argmax(np.abs(ir))

        # Window: keep region around peak, zero out far tail
        # Search for end of IR: where amplitude drops to -60 dB below peak
        peak_amp = np.abs(ir[peak_idx])
        threshold = peak_amp * 10 ** (-60 / 20.0)

        # Find rightmost sample above threshold
        tail_end = len(ir)
        for i in range(peak_idx, len(ir)):
            if np.abs(ir[i]) < threshold:
                tail_end = i
                break

        # Left extent: symmetric window around peak
        window_left = max(0, peak_idx - fade_samples)
        window_right = min(len(ir), tail_end + fade_samples)

        # Apply fade-out ramp at right edge
        if window_right < len(ir):
            fade_len = min(fade_samples, len(ir) - window_right)
            fade_win = np.sin(np.pi * np.arange(fade_len) / fade_len) ** 2
            ir[window_right:window_right + fade_len] *= fade_win[::-1]
        if window_left > 0:
            fade_len = min(fade_samples, window_left)
            fade_win = np.sin(np.pi * np.arange(fade_len) / fade_len) ** 2
            ir[window_left - fade_len:window_left] *= fade_win

        # Zero out beyond window_right
        ir[window_right:] = 0
        if window_left > 0:
            ir[:window_left - fade_len] = 0

        return ir

    def get_hhir(self, recorded: np.ndarray) -> np.ndarray:
        """Get the full Higher Harmonic Impulse Response array (HHIR).

        The HHIR contains overlapping linear + harmonic IRs.
        Use hir_time_position(order) to find where each order is located.
        """
        return self.deconvolve(recorded)

    def hir_time_position(self, order: int, ir_length: int) -> float:
        """Time position (in seconds) of the IR for a given harmonic order.

        Args:
            order: harmonic order (1 = linear/fundamental, 2 = 2nd harmonic, etc.)
            ir_length: length of the IR array in samples.

        Returns:
            time in seconds where this order's IR is centred.
        """
        if order == 1:
            return 0.0
        total_dur = ir_length / self.sample_rate
        return total_dur - self.sweep_period * np.log(order)

    def hir_sample_position(self, order: int, ir_length: int) -> int:
        """Sample index of the IR for a given harmonic order."""
        t = self.hir_time_position(order, ir_length)
        return int(t * self.sample_rate)