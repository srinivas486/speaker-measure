# Audio Engine — Phase 1
# speaker-measure
# Handles device enumeration, playback, capture, and loopback recording
# Supports WASAPI (Windows), ASIO (Windows), and PortAudio (macOS/Linux)

import sounddevice as sd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Host API constants (from PortAudio)
HOSTAPI_MME = 0    # Windows MultiMedia Extensions (legacy)
HOSTAPI_WASAPI = 3 # Windows Audio Session API (preferred for HDMI/AVR multichannel)
HOSTAPI_ASIO = 1   # ASIO (hardware-specific, bypasses Windows audio)
HOSTAPI_WDMKS = 4  # Windows Driver Model (kernel streaming, also good for HDMI)



class AudioDevice:
    """Wrapper for a sounddevice audio device with useful metadata."""

    def __init__(self, info: dict):
        self.id = info["index"]
        self.name = info["name"]
        self.hostapi = info["hostapi"]
        self.hostapi_name = self._get_hostapi_name(info["hostapi"])
        self.max_input_channels = info.get("max_input_channels", 0)
        self.max_output_channels = info.get("max_output_channels", 0)
        self.default_samplerate = info.get("default_samplerate", 48000)
        self.is_asio = "ASIO" in info["name"]
        self.is_usb = "USB" in info["name"]
        self.is_wasapi = self.hostapi_name == "Windows WASAPI"
        self.is_avr = any(k in info["name"].lower() for k in ["hdmi", "avr", "receiver", "denon", "marantz", "yamaha", "onkyo", "pioneer"])

    def _get_hostapi_name(self, hostapi_index: int) -> str:
        try:
            return sd.query_hostapis(hostapi_index)["name"]
        except Exception:
            return "unknown"

    def supports_wasapi(self) -> bool:
        """Return True if this device can be used with WASAPI."""
        return self.is_wasapi and self.max_output_channels >= 2

    def __repr__(self):
        api_tag = f" [{self.hostapi_name}]" if self.hostapi_name != "unknown" else ""
        return f"AudioDevice({self.id}, {self.name!r}{api_tag}, in={self.max_input_channels}, out={self.max_output_channels})"


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

    def list_devices_by_hostapi(self, hostapi_name_fragment: str) -> list[AudioDevice]:
        """Return devices matching a host API (e.g. 'wasapi', 'asio', 'mme'). Case-insensitive."""
        fragment = hostapi_name_fragment.lower()
        return [d for d in self.list_devices() if fragment in d.hostapi_name.lower()]

    def list_playback_devices(self) -> list[AudioDevice]:
        """Return devices that have output channels."""
        return [d for d in self.list_devices() if d.max_output_channels > 0]

    def list_playback_devices_by_hostapi(self, hostapi_name_fragment: str) -> list[AudioDevice]:
        """Return output devices matching a host API."""
        fragment = hostapi_name_fragment.lower()
        return [d for d in self.list_playback_devices() if fragment in d.hostapi_name.lower()]

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

    def query_device_info(self, device_id: int) -> dict:
        """Return full info dict for a device via sounddevice."""
        return sd.query_devices(device_id)

    def get_playback_channel_count(self, device: AudioDevice) -> int:
        """Return max output channels for a device."""
        return device.max_output_channels

    def auto_pick_single_mic(self) -> Optional[AudioDevice]:
        """Return mic if exactly one capture device exists, otherwise None."""
        caps = self.list_capture_devices()
        return caps[0] if len(caps) == 1 else None

    def auto_pick_hdmi_or_first_playback(self) -> Optional[AudioDevice]:
        """Return HDMI device if found, otherwise first playback device."""
        hdmi = self.find_hdmi_audio()
        if hdmi:
            return hdmi
        pbs = self.list_playback_devices()
        return pbs[0] if pbs else None

    def interactive_pick_devices(self) -> tuple[AudioDevice, AudioDevice, int]:
        """Interactive device picker: playback first, then capture.

        Auto-selects capture if only one mic available.
        Auto-selects HDMI (or first playback) as default.
        Shows host API (WASAPI/ASIO/MME) for each device.
        """
        print("\n=== Playback Devices ===")
        pbs = self.list_playback_devices()
        default_pb = self.auto_pick_hdmi_or_first_playback()
        for dev in pbs:
            api_tag = f" [{dev.hostapi_name}]" if dev.hostapi_name != "unknown" else ""
            marker = " (default)" if default_pb and dev.id == default_pb.id else ""
            print(f"  [{dev.id}] {dev.name} ({dev.max_output_channels}ch){api_tag}{marker}")

        print("\n=== Capture Devices ===")
        caps = self.list_capture_devices()
        # Prefer UMIK or any professional/calibration mic as default
        default_cap = next(
            (d for d in caps if any(k in d.name.lower() for k in ["umik", "cal", "calibration", "measurement", "mic"])),
            self.auto_pick_single_mic(),
        )
        for dev in caps:
            marker = " (auto-selected — only mic)" if default_cap and dev.id == default_cap.id else ""
            print(f"  [{dev.id}] {dev.name} ({dev.max_input_channels}ch){marker}")

        # Playback: default to HDMI/first
        pb_id = input(f"Playback device [{default_pb.id} if default, Enter to confirm]: ").strip()
        pb_id = int(pb_id) if pb_id else (default_pb.id if default_pb else None)
        pb = next((d for d in self.list_devices() if d.id == pb_id), default_pb)

        # Capture: default to single mic, or ask
        if default_cap:
            cap = default_cap
            print(f"Capture device auto-selected: {cap.name}")
        else:
            cap_id_str = input("Capture device (id): ").strip()
            cap_id = int(cap_id_str) if cap_id_str else None
            cap = next((d for d in self.list_devices() if d.id == cap_id), None)
            if not cap:
                raise RuntimeError("No capture device selected")

        # Show detected layout for selected playback device
        ch_list = self.build_channel_list_for_device(pb, num_subwoofers=0)
        layout_channels = [c for c, _, is_sub in ch_list if not is_sub]
        layout_subs = [c for c, _, is_sub in ch_list if is_sub]
        print(f"  Detected layout ({pb.max_output_channels}ch): {layout_channels}")
        if layout_subs:
            print(f"  Subwoofer channel(s): {layout_subs}")

        # Subwoofer config if LFE channel exists
        num_sub = 0
        if layout_subs:
            num_sub_str = input(f"Number of subwoofers to measure [1]: ").strip()
            num_sub = int(num_sub_str) if num_sub_str else 1
            num_sub = min(num_sub, len(layout_subs))

        return pb, cap, num_sub

    def build_channel_list_for_device(
        self,
        device: AudioDevice,
        num_subwoofers: int = 0,
    ) -> list[tuple[str, int, bool]]:
        """Map a device channel count to (channel_id, hdmi_index, is_subwoofer) tuples."""
        n = device.max_output_channels
        layout_map = {
            2:  [("FL", 0, False), ("FR", 1, False)],
            6:  [("FL", 0, False), ("C", 1, False), ("FR", 2, False),
                 ("BL", 3, False), ("BR", 4, False), ("LFE", 5, True)],
            8:  [("FL", 0, False), ("C", 1, False), ("FR", 2, False),
                 ("BL", 3, False), ("BR", 4, False),
                 ("SL", 5, False), ("SR", 6, False), ("LFE", 7, True)],
            10: [("FL", 0, False), ("C", 1, False), ("FR", 2, False),
                 ("FHL", 3, False), ("FHR", 4, False),
                 ("BL", 5, False), ("BR", 6, False),
                 ("SL", 7, False), ("SR", 8, False), ("LFE", 9, True)],
            12: [("FL", 0, False), ("C", 1, False), ("FR", 2, False),
                 ("SRL", 3, False), ("SRR", 4, False),
                 ("RL", 5, False), ("RR", 6, False),
                 ("FTL", 7, False), ("FTR", 8, False),
                 ("TFL", 9, False), ("TFR", 10, False), ("LFE", 11, True)],
            14: [("FL", 0, False), ("C", 1, False), ("FR", 2, False),
                 ("SRL", 3, False), ("SRR", 4, False),
                 ("RL", 5, False), ("RR", 6, False),
                 ("FTL", 7, False), ("FTR", 8, False),
                 ("TFL", 9, False), ("TFR", 10, False),
                 ("SW1", 11, True), ("SW2", 12, True), ("LFE", 13, False)],
        }
        if n not in layout_map:
            channels = []
            for i in range(n):
                if i == 0:
                    channels.append(("FL", i, False))
                elif i == 1:
                    channels.append(("FR", i, False))
                elif i == n - 1:
                    channels.append(("LFE", i, True))
                else:
                    channels.append((f"CH{i}", i, False))
            return channels
        channels = layout_map[n]
        non_sub = [(c, idx, is_sub) for c, idx, is_sub in channels if not is_sub]
        subs = [(c, idx, is_sub) for c, idx, is_sub in channels if is_sub]
        if num_subwoofers > 0:
            included_subs = subs[:num_subwoofers]
            return non_sub + included_subs
        return non_sub + subs

    def find_wasapi_hdmi(self) -> Optional[AudioDevice]:
        """Find HDMI/AVR audio via WASAPI (Windows Audio Session API).

        WASAPI is preferred for multichannel HDMI/AVR because:
        - Supports exclusive mode for bit-perfect multichannel PCM
        - Properly exposes all HDMI channel beds (5.1/7.1/Atmos)
        - Lower latency than MME on Windows
        - Works with most AVR/receivers via HDMI
        """
        candidates = []
        for dev in self.list_playback_devices():
            name = dev.name.lower()
            is_hdmi_avr = any(k in name for k in ["hdmi", "avr", "receiver", "denon", "marantz", "yamaha", "onkyo", "pioneer"])
            is_wasapi = dev.is_wasapi
            if is_hdmi_avr and is_wasapi:
                candidates.append(dev)

        if candidates:
            logger.info(f"WASAPI HDMI/AVR candidates: {[d.name for d in candidates]}")
            return candidates[0]

        # Fallback: WASAPI device with many channels even if not named HDMI/AVR
        for dev in self.list_playback_devices():
            if dev.is_wasapi and dev.max_output_channels >= 4:
                candidates.append(dev)

        if candidates:
            logger.info(f"WASAPI multichannel fallback: {[d.name for d in candidates]}")
            return candidates[0]

        return None

    def find_asio_hdmi(self) -> Optional[AudioDevice]:
        """Find HDMI/AVR audio via ASIO (less common — some systems use ASIO for HDMI too)."""
        candidates = []
        for dev in self.list_playback_devices():
            name = dev.name.lower()
            is_hdmi_avr = any(k in name for k in ["hdmi", "avr", "receiver", "denon", "marantz", "yamaha", "onkyo", "pioneer"])
            if is_hdmi_avr and dev.is_asio:
                candidates.append(dev)

        if candidates:
            logger.info(f"ASIO HDMI/AVR candidates: {[d.name for d in candidates]}")
            return candidates[0]

        # ASIO multichannel fallback (some ASIO interfaces are multichannel)
        for dev in self.list_playback_devices():
            if dev.is_asio and dev.max_output_channels >= 4:
                candidates.append(dev)

        return candidates[0] if candidates else None

    def find_hdmi_audio(self) -> Optional[AudioDevice]:
        """Find HDMI audio output (AVR or GPU HDMI audio).

        Priority:
        1. WASAPI HDMI/AVR (recommended for Windows — proper multichannel + Atmos)
        2. ASIO HDMI/AVR (if WASAPI not available)
        3. Any HDMI-named device (MME fallback)
        """
        # Try WASAPI first — it's the modern Windows audio API and works best with AVRs
        wasapi = self.find_wasapi_hdmi()
        if wasapi:
            return wasapi

        # Fall back to ASIO if WASAPI isn't available
        asio = self.find_asio_hdmi()
        if asio:
            logger.warning("Using ASIO for HDMI audio — WASAPI preferred. If you have issues, check ASIO drivers.")
            return asio

        # Last resort: any HDMI-named device via MME
        candidates = []
        for dev in self.list_playback_devices():
            name = dev.name.lower()
            if any(k in name for k in ["hdmi", "avr", "receiver", "nvidia", "amd", "intel"]):
                candidates.append(dev)
        return candidates[0] if candidates else None

    def find_hdmi_audio_with_preference(self, preferred_api: str = "wasapi") -> Optional[AudioDevice]:
        """Find HDMI audio preferring a specific API.

        Args:
            preferred_api: "wasapi", "asio", or "any" (default "wasapi")

        Returns:
            AudioDevice or None.
        """
        if preferred_api == "wasapi":
            return self.find_wasapi_hdmi() or self.find_asio_hdmi() or self.find_hdmi_audio()
        elif preferred_api == "asio":
            return self.find_asio_hdmi() or self.find_wasapi_hdmi() or self.find_hdmi_audio()
        else:  # "any"
            return self.find_hdmi_audio()

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