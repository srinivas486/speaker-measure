# Audio Engine — Phase 1
# speaker-measure
# Handles device enumeration, playback, capture, and loopback recording

import sounddevice as sd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class AudioDevice:
    """Wrapper for a sounddevice audio device with useful metadata."""

    def __init__(self, info: dict):
        self.id = info["index"]
        self.name = info["name"]
        self.hostapi = info["hostapi"]
        self.max_input_channels = info.get("max_input_channels", 0)
        self.max_output_channels = info.get("max_output_channels", 0)
        self.default_samplerate = info.get("default_samplerate", 48000)
        self.is_asio = "ASIO" in info["name"]
        self.is_usb = "USB" in info["name"]

    def __repr__(self):
        return f"AudioDevice({self.id}, {self.name!r}, in={self.max_input_channels}, out={self.max_output_channels})"


class AudioEngine:
    """Low-level audio I/O using sounddevice (PortAudio + WASAPI/ASIO).

    Signal flow:
      - Playback: sweep WAV → selected playback device (e.g. HDMI/AVR via WASAPI or ASIO)
      - Capture: UMIK-1 (USB mic) → selected input device
      - Use sd.sleep() for buffer-friendly loop in blocking mode
    """

    # Standard measurement sample rate
    DEFAULT_SAMPLE_RATE = 48000
    STANDARD_SAMPLE_RATES = [44100, 48000, 96000]

    # Buffer sizes for low-latency playback/recording
    BUFFER_SIZES = [256, 512, 1024, 2048, 4096]

    def __init__(self):
        self.playback_device: Optional[AudioDevice] = None
        self.capture_device: Optional[AudioDevice] = None
        self.sample_rate: int = self.DEFAULT_SAMPLE_RATE
        self.buffer_size: int = 1024
        self._stream: Optional[sd.Stream] = None

    # -------------------------------------------------------------------------
    # Device enumeration
    # -------------------------------------------------------------------------

    def list_devices(self) -> list[AudioDevice]:
        """Return all audio devices as AudioDevice objects."""
        infos = sd.query_devices()
        if isinstance(infos, dict):
            infos = [infos]
        return [AudioDevice(info) for info in infos]

    def list_playback_devices(self) -> list[AudioDevice]:
        """Return devices that have output channels."""
        return [d for d in self.list_devices() if d.max_output_channels > 0]

    def list_capture_devices(self) -> list[AudioDevice]:
        """Return devices that have input channels (e.g. UMIK-1 USB mic)."""
        return [d for d in self.list_devices() if d.max_input_channels > 0]

    def find_device_by_name(self, name_fragment: str, prefer_hostapi: Optional[int] = None) -> Optional[AudioDevice]:
        """Find a device whose name contains the given fragment."""
        candidates = self.list_devices()
        if prefer_hostapi is not None:
            candidates = [d for d in candidates if d.hostapi == prefer_hostapi]
        for dev in candidates:
            if name_fragment.lower() in dev.name.lower():
                return dev
        return None

    def find_umik1(self) -> Optional[AudioDevice]:
        """Find UMIK-1 USB measurement mic by name pattern."""
        return self.find_device_by_name("umik") or self.find_device_by_name("usb audio")

    def find_hdmi_audio(self) -> Optional[AudioDevice]:
        """Find HDMI audio output (AVR or GPU HDMI audio)."""
        candidates = []
        for dev in self.list_playback_devices():
            name = dev.name.lower()
            if any(k in name for k in ["hdmi", "avr", "receiver", "nvidia", "amd", "intel"]):
                candidates.append(dev)
        # Prefer non-ASIO (WASAPI) for HDMI
        non_asio = [d for d in candidates if not d.is_asio]
        return non_asio[0] if non_asio else candidates[0] if candidates else None

    # -------------------------------------------------------------------------
    # Device configuration
    # -------------------------------------------------------------------------

    def set_playback_device(self, device: AudioDevice) -> None:
        self.playback_device = device
        logger.info(f"Playback device set: {device}")

    def set_capture_device(self, device: AudioDevice) -> None:
        self.capture_device = device
        logger.info(f"Capture device set: {device}")

    def set_sample_rate(self, rate: int) -> None:
        if rate not in self.STANDARD_SAMPLE_RATES:
            raise ValueError(f"Unsupported sample rate: {rate}. Use one of {self.STANDARD_SAMPLE_RATES}")
        self.sample_rate = rate
        logger.info(f"Sample rate set: {rate} Hz")

    def set_buffer_size(self, size: int) -> None:
        if size not in self.BUFFER_SIZES:
            raise ValueError(f"Invalid buffer size: {size}. Use one of {self.BUFFER_SIZES}")
        self.buffer_size = size
        logger.info(f"Buffer size set: {size}")

    # -------------------------------------------------------------------------
    # Stream management
    # -------------------------------------------------------------------------

    def open_stream(
        self,
        channels_playback: int = 2,
        channels_capture: int = 1,
    ) -> sd.Stream:
        """Open a combined playback+capture stream.

        Args:
            channels_playback: Number of output channels (2 = stereo, 8 = 7.1, etc.)
            channels_capture: Number of input channels (1 = mono mic, 2 = stereo line-in)

        Returns:
            An sd.Stream instance (started on open).
        """
        if self._stream is not None:
            self.close_stream()

        self._stream = sd.Stream(
            device=(self.playback_device.id if self.playback_device else None,
                    self.capture_device.id if self.capture_device else None),
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            channels=(channels_playback, channels_capture),
            dtype="float32",
            latency="low",
        )
        self._stream.start()
        logger.info(f"Stream opened: playback={channels_playback}ch, capture={channels_capture}ch, "
                    f"sr={self.sample_rate}, buffer={self.buffer_size}")
        return self._stream

    def close_stream(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            logger.info("Stream closed")

    def is_stream_active(self) -> bool:
        return self._stream is not None and self._stream.active

    # -------------------------------------------------------------------------
    # Playback + Capture helpers
    # -------------------------------------------------------------------------

    def play_capture_buffers(
        self,
        audio_data: np.ndarray,
        num_frames: int,
        blocksize: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Play audio through playback device and simultaneously record from capture device.

        Uses sd.playrec() for synchronized play+rec. This is the recommended approach
        for measurement sweeps where we need precise timing alignment between
        what was played and what was recorded.

        Args:
            audio_data: The audio to play. Shape (n_frames, n_playback_channels).
            num_frames: Number of frames to capture (same as audio_data frames).
            blocksize: Internal blocksize for the stream.

        Returns:
            Tuple of (played_audio, captured_audio) as numpy arrays.
        """
        if self._stream is None:
            raise RuntimeError("Stream not open. Call open_stream() first.")

        # Use sd.playrec for simultaneous play+rec
        recorded = np.zeros((num_frames, self._stream.channels[1]), dtype=np.float32)
        played = np.zeros((num_frames, self._stream.channels[0]), dtype=np.float32)

        with self._stream:
            record_buf = recorded
            # Write audio in blocks and read capture simultaneously
            pos = 0
            block_size = blocksize
            while pos < num_frames:
                end = min(pos + block_size, num_frames)
                block = audio_data[pos:end]
                # Pad block if short
                if block.shape[0] < block_size:
                    block = np.pad(block, ((0, block_size - block.shape[0]), (0, 0)), mode="constant")
                try:
                    recorded_chunk = self._stream.read(block_size)
                    # recorded_chunk is (recorded_data, status)
                    if isinstance(recorded_chunk, tuple):
                        recorded_chunk = recorded_chunk[0]
                    record_buf[pos:min(pos + len(recorded_chunk), num_frames)] = recorded_chunk[:num_frames - pos] if len(recorded_chunk) >= num_frames - pos else recorded_chunk
                except Exception as e:
                    logger.warning(f"Read error at pos {pos}: {e}")

                # Play block
                self._stream.write(block)
                pos += block_size

        return played, record_buf

    # -------------------------------------------------------------------------
    # Simple playback (for testing)
    # -------------------------------------------------------------------------

    def play_wav(self, wav_path: str | Path, blocking: bool = True) -> None:
        """Play a WAV file through the selected playback device."""
        import soundfile as sf
        data, sr = sf.read(str(wav_path), dtype="float32")
        if sr != self.sample_rate:
            raise ValueError(f"WAV sample rate {sr} does not match engine sample rate {self.sample_rate}")
        sd.play(data, device=self.playback_device.id, samplerate=sr, blocking=blocking)
        logger.info(f"Played: {wav_path}")

    def record(self, duration_sec: float) -> np.ndarray:
        """Record from the capture device for the given duration."""
        frames = int(duration_sec * self.sample_rate)
        data = sd.rec(frames, device=self.capture_device.id, samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        return data

    # -------------------------------------------------------------------------
    # Latency info
    # -------------------------------------------------------------------------

    def get_latency_ms(self) -> dict:
        """Return estimated latency for current device configuration."""
        if self._stream is None:
            return {"playback_ms": None, "capture_ms": None, "roundtrip_ms": None}

        # sounddevice reports latency in seconds
        return {
            "playback_ms": round(self._stream.latency[0] * 1000, 1) if self._stream.latency[0] else None,
            "capture_ms": round(self._stream.latency[1] * 1000, 1) if self._stream.latency[1] else None,
            "roundtrip_ms": round(sum(self._stream.latency) * 1000, 1) if all(self._stream.latency) else None,
        }


# -----------------------------------------------------------------------------
# Dev helper / test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = AudioEngine()

    print("\n=== All Devices ===")
    for dev in engine.list_devices():
        print(f"  [{dev.id}] {dev.name} | in={dev.max_input_channels} out={dev.max_output_channels} | hostapi={dev.hostapi}")

    print("\n=== Playback Devices ===")
    for dev in engine.list_playback_devices():
        print(f"  [{dev.id}] {dev.name} ({dev.max_output_channels}ch)")

    print("\n=== Capture Devices ===")
    for dev in engine.list_capture_devices():
        print(f"  [{dev.id}] {dev.name} ({dev.max_input_channels}ch)")

    print("\n=== Auto-detected ===")
    umik = engine.find_umik1()
    hdmi = engine.find_hdmi_audio()
    print(f"  UMIK-1: {umik}")
    print(f"  HDMI:   {hdmi}")