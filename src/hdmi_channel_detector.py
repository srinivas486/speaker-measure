# HDMI / AVR Channel Detector
# speaker-measure
# Detects all available output channels from HDMI/ASIO audio devices,
# maps them to standard speaker labels, and reports subwoofer count.

import sounddevice as sd
import re
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# Standard channel label mapping (ITU-R BS.2159-7 / Dolby/AES standards)
# Positional index within HDMI multichannel stream → label
STANDARD_CHANNEL_MAP = [
    "FL",   # 0 — Front Left
    "FR",   # 1 — Front Right
    "C",    # 2 — Center
    "LFE",  # 3 — Low Frequency Effects (subwoofer)
    "BL",   # 4 — Back Left (or Surround Left)
    "BR",   # 5 — Back Right (or Surround Right)
    "FLc",  # 6 — Front Left Centre (height)
    "FRc",  # 7 — Front Right Centre (height)
    "BC",   # 8 — Back Centre
    "SL",   # 9 — Surround Left
    "SR",   # 10 — Surround Right
    "SW1",  # 11 — Subwoofer 1 (when multiple subwoofers are configured)
    "SW2",  # 12 — Subwoofer 2
    "SW3",  # 13 — Subwoofer 3
    "SW4",  # 14 — Subwoofer 4
    "FHL",  # 15 — Front Height Left
    "FHR",  # 16 — Front Height Right
    "FVL",  # 17 — Front V现实 Height Left (not standard)
    "FVR",  # 18 — Front V现实 Height Right
    "SDL",  # 19 — Surround Back Left
    "SDR",  # 20 — Surround Back Right
    "CS",   # 21 — Centre Surround (back centre)
    "MS",   # 22 — Middle Surround
    "UD",   # 23 — Upward Dismiss (various height standards)
    # === Additional Atmos/Auro 3D channels ===
    "FDL",  # 24 — Front Dolby Left (Atmos)
    "FDR",  # 25 — Front Dolby Right (Atmos)
    "SDL2", # 26 — Surround Dolby Left (Atmos)
    "SDR2", # 27 — Surround Dolby Right (Atmos)
    "VHL",  # 28 — Vertical Height Left
    "VHR",  # 29 — Vertical Height Right
    "TS",   # 30 — Top Surround
    "BT",   # 31 — Back Top
]


@dataclass
class HdmiChannelInfo:
    """Represents a detected HDMI/ASIO output channel."""
    index: int                    # Channel index within the device stream (0-based)
    label: str                    # Standard speaker label (e.g. "FL", "SW1")
    is_subwoofer: bool           # True if this is an LFE or SWx channel
    sub_index: Optional[int]      # For multiple subwoofers: 1, 2, 3, or None
    frequency_hz: Optional[int]  # Recommended crossover Hz (None = full range)


@dataclass
class HdmiDeviceInfo:
    """Full HDMI/ASIO audio device with all channel information."""
    device_index: int
    name: str
    hostapi: str
    channel_count: int           # Total number of output channels
    channels: list[HdmiChannelInfo] = field(default_factory=list)
    subwoofer_count: int = 0      # How many subwoofer channels detected
    layout_type: str = ""         # e.g. "7.1", "5.1.2", "9.4.6", "unknown"


class HdmiChannelDetector:
    """Queries audio devices and maps their output channels to speaker labels.

    On Windows with ASIO, HDMI devices (AVR receivers) typically expose
    multichannel PCM streams (8ch = 7.1, 16ch = 9.4.6, etc.).
    The channel ordering follows ITU-R BS.2159-7 / Dolby standards.

    Usage:
        detector = HdmiChannelDetector()
        devices = detector.list_hdmi_devices()
        info = detector.detect_channels(device_index)
        print(info.layout_type, info.subwoofer_count, info.channels)
    """

    def __init__(self):
        self._cache: dict[int, HdmiDeviceInfo] = {}

    # -------------------------------------------------------------------------
    # Device enumeration
    # -------------------------------------------------------------------------

    def list_hdmi_devices(self) -> list[HdmiDeviceInfo]:
        """List all HDMI/AVR audio output devices (WASAPI or ASIO).

        Returns:
            List of HdmiDeviceInfo objects for devices with HDMI/AVR in their name.
        """
        devices = []
        all_devs = sd.query_devices()
        if isinstance(all_devs, dict):
            all_devs = [all_devs]

        for dev in all_devs:
            name = dev["name"].lower()
            if any(k in name for k in ["hdmi", "avr", "receiver", "denon", "marantz", "yamaha"]):
                if dev["max_output_channels"] >= 2:
                    info = self._build_device_info(dev)
                    devices.append(info)

        return devices

    def list_all_multichannel_devices(self) -> list[HdmiDeviceInfo]:
        """List ALL output devices with 4+ channels (not just HDMI-named).

        Use this as a fallback to find any multichannel audio interface.
        """
        devices = []
        all_devs = sd.query_devices()
        if isinstance(all_devs, dict):
            all_devs = [all_devs]

        for dev in all_devs:
            if dev["max_output_channels"] >= 4:
                info = self._build_device_info(dev)
                devices.append(info)

        return devices

    def detect_channels(self, device_index: int) -> HdmiDeviceInfo:
        """Detect and map all channels for a given device index.

        Args:
            device_index: sounddevice device index (from query_devices).

        Returns:
            HdmiDeviceInfo with full channel mapping.
        """
        if device_index in self._cache:
            return self._cache[device_index]

        try:
            dev = sd.query_devices(device_index)
        except sd.PortAudioError:
            logger.error(f"Device {device_index} not found")
            return HdmiDeviceInfo(device_index=device_index, name="unknown", hostapi="", channel_count=0)

        info = self._build_device_info(dev)
        self._cache[device_index] = info
        return info

    # -------------------------------------------------------------------------
    # Channel mapping logic
    # -------------------------------------------------------------------------

    def _build_device_info(self, dev: dict) -> HdmiDeviceInfo:
        """Build HdmiDeviceInfo from a query_devices dict."""
        ch_count = dev["max_output_channels"]
        name = dev["name"]
        hostapi = sd.query_hostapis(dev["hostapi"])["name"]

        channels = self._map_channels(ch_count)
        sub_count = sum(1 for c in channels if c.is_subwoofer)

        layout = self._derive_layout(channels, ch_count)

        return HdmiDeviceInfo(
            device_index=dev["index"],
            name=name,
            hostapi=hostapi,
            channel_count=ch_count,
            channels=channels,
            subwoofer_count=sub_count,
            layout_type=layout,
        )

    def _map_channels(self, channel_count: int) -> list[HdmiChannelInfo]:
        """Map channel indices 0..N to standard speaker labels.

        HDMI multichannel ordering (Dolby/DTS standard):
          0=FL, 1=FR, 2=C, 3=LFE, 4=BL, 5=BR, 6=FLc, 7=FRc, 8=BC, 9=SL, 10=SR
          11=SW1, 12=SW2, 13=SW3, 14=SW4

        When HDMI reports LFE as channel 3 but the AVR has 2 or 4 subwoofers,
        the remaining subwoofer channels (SW2, SW3, SW4) are at indices 11, 12, 13.
        """
        channels = []
        for idx in range(channel_count):
            if idx < len(STANDARD_CHANNEL_MAP):
                label = STANDARD_CHANNEL_MAP[idx]
            else:
                label = f"CH{idx}"

            is_sub = self._is_subwoofer_label(label)
            sub_index = self._extract_sub_index(label) if is_sub else None

            channels.append(HdmiChannelInfo(
                index=idx,
                label=label,
                is_subwoofer=is_sub,
                sub_index=sub_index,
                frequency_hz=None,
            ))

        return channels

    def _is_subwoofer_label(self, label: str) -> bool:
        """Returns True for LFE or SW1/SW2/SW3/SW4 labels."""
        return label in ("LFE",) or label.startswith("SW")

    def _extract_sub_index(self, label: str) -> Optional[int]:
        """Extract subwoofer number from SW1/SW2/SW3/SW4, or 1 for plain LFE."""
        if label == "LFE":
            return 1
        m = re.search(r"SW(\d)", label)
        return int(m.group(1)) if m else None

    def _derive_layout(self, channels: list[HdmiChannelInfo], total: int) -> str:
        """Derive a human-readable layout string like '7.1', '5.1.2', '9.4.6'."""
        non_sub = [c for c in channels if not c.is_subwoofer]
        sub_count = len([c for c in channels if c.is_subwoofer])

        # Count bed channels (standard 7.1 layout = 8 channels, bed is 5.1 or 7.1)
        bed_channels = len(non_sub)

        # Heuristic: typical layouts
        if total == 2:
            return "2.0"
        elif total == 4:
            return "3.1" if sub_count >= 1 else "4.0"
        elif total == 6:
            return "5.1"
        elif total == 8:
            # 7.1 bed
            return "7.1"
        elif total == 10:
            # 7.1.2 Atmos (7.1 bed + 2 height)
            height_count = self._count_height_channels(channels)
            return f"7.1.{height_count}" if height_count else "7.1"
        elif total == 12:
            # Could be 7.1.4 or 9.1 Atmos
            height_count = self._count_height_channels(channels)
            if height_count >= 2:
                return f"7.1.{height_count}"
            return "9.1"
        elif total == 16:
            # 9.4.6 Atmos full
            return "9.4.6"
        else:
            return f"{total}ch"

    def _count_height_channels(self, channels: list[HdmiChannelInfo]) -> int:
        """Count obvious height/Atmos channels."""
        height_kws = {"hl", "hr", "fdl", "fdr", "sdl", "sdr", "vh", "bt", "ts", "cs", "ud"}
        return sum(1 for c in channels if any(k in c.label.lower() for k in height_kws))

    # -------------------------------------------------------------------------
    # Subwoofer-specific helpers
    # -------------------------------------------------------------------------

    def get_all_subwoofer_channels(self, device_info: HdmiDeviceInfo) -> list[HdmiChannelInfo]:
        """Return all subwoofer channels (LFE + SW1/SW2/SW3/SW4) from a device."""
        return [c for c in device_info.channels if c.is_subwoofer]

    def get_subwoofer_by_index(
        self,
        device_info: HdmiDeviceInfo,
        sub_index: int,
    ) -> Optional[HdmiChannelInfo]:
        """Get the channel info for subwoofer number N (1-indexed).

        Note: HDMI reports LFE as a single channel. When there are 2+ physical
        subwoofers, the AVR labels all of them as LFE in the HDMI stream, and
        they are distinguished via the SW1/SW2/SW3/SW4 labels at indices 11+.

        Args:
            device_info: HdmiDeviceInfo from detect_channels()
            sub_index: 1 for first sub, 2 for second, etc.

        Returns:
            HdmiChannelInfo or None if that sub index doesn't exist.
        """
        subs = self.get_all_subwoofer_channels(device_info)
        for c in subs:
            if c.sub_index == sub_index:
                return c
        return None

    def has_multiple_subwoofers(self, device_info: HdmiDeviceInfo) -> bool:
        """True if the device has 2 or more subwoofer channels."""
        return device_info.subwoofer_count >= 2


# -----------------------------------------------------------------------------
# Convenience function for use by other modules
# -----------------------------------------------------------------------------

def detect_avr_channels(
    device_index: Optional[int] = None,
    device_name_fragment: Optional[str] = None,
) -> HdmiDeviceInfo:
    """One-shot detection of AVR/HDMI channels.

    Provide either device_index (sounddevice id) or device_name_fragment
    (e.g. "HDMI" or "Denon"). If neither, auto-detects first multichannel device.

    Returns HdmiDeviceInfo.
    """
    detector = HdmiChannelDetector()

    if device_index is not None:
        return detector.detect_channels(device_index)

    if device_name_fragment:
        all_devs = sd.query_devices()
        if isinstance(all_devs, dict):
            all_devs = [all_devs]
        for dev in all_devs:
            if device_name_fragment.lower() in dev["name"].lower():
                return detector.detect_channels(dev["index"])

    # Auto: find first multichannel device
    devices = detector.list_all_multichannel_devices()
    if not devices:
        raise RuntimeError("No multichannel HDMI/AVR audio device found")
    return devices[0]


# -----------------------------------------------------------------------------
# CLI / test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    logging.basicConfig(level=logging.INFO)
    detector = HdmiChannelDetector()

    print("\n=== HDMI / AVR Devices ===")
    hdmi_devs = detector.list_hdmi_devices()
    for d in hdmi_devs:
        print(f"  [{d.device_index}] {d.name} ({d.hostapi}) — {d.channel_count}ch — {d.layout_type}")

    print("\n=== All Multichannel Devices (fallback) ===")
    mc_devs = detector.list_all_multichannel_devices()
    for d in mc_devs:
        print(f"  [{d.device_index}] {d.name} ({d.hostapi}) — {d.channel_count}ch — {d.layout_type}")

    if hdmi_devs:
        info = hdmi_devs[0]
    elif mc_devs:
        info = detector.detect_channels(mc_devs[0].device_index)
    else:
        print("\nNo multichannel devices found.")
        info = None

    if info:
        print(f"\n=== Channel Map: {info.name} ({info.layout_type}) ===")
        print(f"{'Idx':<5} {'Label':<8} {'Type':<12} {'Sub#'}")
        print("-" * 35)
        for ch in info.channels:
            sub_str = str(ch.sub_index) if ch.sub_index else "—"
            typ = "subwoofer" if ch.is_subwoofer else "speaker"
            print(f"{ch.index:<5} {ch.label:<8} {typ:<12} {sub_str}")

        print(f"\nTotal subwoofers: {info.subwoofer_count}")
        print(f"Has multiple subwoofers: {detector.has_multiple_subwoofers(info)}")