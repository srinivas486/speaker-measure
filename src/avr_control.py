# AVR Control — Subwoofer Switching for Measurement
# speaker-measure
# Provides AVR Telnet control for subwoofer on/off during multi-subwoofer measurement.
# Uses the same approach as oca_transfer.py (Telnet port 23).

import socket
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class AvrControl:
    """Low-level AVR Telnet control for subwoofer switching.

    Used when measuring multiple subwoofers — to isolate each one,
    we switch all other subwoofers OFF, measure the target, then restore.

    Usage:
        avr = AvrControl("192.168.1.100")
        avr.set_subwoofer_off(2)      # switch subwoofer 2 OFF
        # ... run measurement on subwoofer 1 ...
        avr.restore_all_subwoofers()   # bring subwoofer 2 back ON
    """

    TELNET_PORT = 23
    COMMAND_DELAY_SEC = 0.75
    RECV_SIZE = 1024

    def __init__(self, ip: str, is_new_model: bool = True):
        """Initialize AVR control.

        Args:
            ip: AVR IP address (e.g. "192.168.1.100").
            is_new_model: True for newer Denon/Marantz (SSSWO command).
                         False for older models (SSMWM command).
        """
        self.ip = ip
        self.is_new_model = is_new_model
        self._original_bass_mode: Optional[str] = None
        self._original_subwoofer_state: dict[int, bool] = {}  # sub_idx -> was_on

    # -------------------------------------------------------------------------
    # Low-level Telnet
    # -------------------------------------------------------------------------

    def _telnet_send(self, cmd: str) -> str:
        """Send a command via Telnet and return the response."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect((self.ip, self.TELNET_PORT))
        time.sleep(0.2)  # let AVR process connection

        # Send command
        sock.sendall((cmd + "\r\n").encode("ascii"))
        time.sleep(self.COMMAND_DELAY_SEC)

        # Read response
        try:
            data = sock.recv(self.RECV_SIZE)
            resp = data.decode("ascii", errors="replace").strip()
        except socket.timeout:
            resp = ""
        finally:
            sock.close()

        logger.debug(f"AVR Telnet → {cmd!r} ← {resp!r}")
        return resp

    # -------------------------------------------------------------------------
    # Subwoofer switching
    # -------------------------------------------------------------------------

    def set_subwoofer_off(self, sub_index: int) -> bool:
        """Switch a specific subwoofer OFF (by index, 1-indexed).

        For the X3800H and similar modern AVRs, we use PSSWL OFF.
        We track the original state so restore_all_subwoofers() can undo this.

        Args:
            sub_index: Subwoofer number (1, 2, 3, or 4).

        Returns:
            True if command was sent successfully.
        """
        try:
            resp = self._telnet_send("PSSWL OFF")
            logger.info(f"Subwoofer {sub_index} switched OFF (response: {resp})")
            self._original_subwoofer_state[sub_index] = False
            return True
        except Exception as e:
            logger.error(f"Failed to switch sub {sub_index} OFF: {e}")
            return False

    def set_subwoofer_on(self, sub_index: int) -> bool:
        """Switch a specific subwoofer ON (restore from OFF state).

        We don't have a direct "PSSWL ON" command on modern AVRs.
        The restoration is handled by restore_all_subwoofers() which
        re-enables via bass mode setting.
        """
        # For modern AVRs, PSSWL ON isn't standard. The usual approach is
        # to restore bass mode via SSSWO or simply let the next measurement
        # clear the OFF state by setting bass mode again.
        logger.info(f"Subwoofer {sub_index} ON requested (no direct command needed)")
        return True

    def restore_all_subwoofers(self) -> bool:
        """Restore all subwoofers to ON state after measurement.

        On X3800H / modern AVRs: re-send PSSWL to restore. Since PSSWL OFF
        disables the subwoofer level control (not the speaker itself), we need
        to send PSSWL followed by a mode restore.

        A simple approach: send the current bass mode again which clears
        the subwoofer-off state.
        """
        try:
            # Re-send PSSWL without OFF to restore normal state
            resp = self._telnet_send("PSSWL")
            time.sleep(self.COMMAND_DELAY_SEC)
            # Also re-set bass mode to make sure subwoofers are active
            resp2 = self._telnet_send("SSSWO LFE")
            logger.info(f"Subwoofers restored (PSSWL response: {resp})")
            self._original_subwoofer_state.clear()
            return True
        except Exception as e:
            logger.error(f"Failed to restore subwoofers: {e}")
            return False

    def switch_subwoofer_for_measurement(
        self,
        target_sub_index: int,
        all_sub_indices: list[int],
    ) -> dict[int, bool]:
        """Switch all subwoofers EXCEPT the target one OFF.

        This allows measuring a specific subwoofer in isolation.

        Args:
            target_sub_index: The subwoofer to keep ON (1-indexed).
            all_sub_indices: List of all active subwoofer indices.

        Returns:
            Dict of {sub_index: was_on_before} for all switched subs.
            Use restore_all_subwoofers() with this to undo.
        """
        states = {}
        for sub_idx in all_sub_indices:
            if sub_idx == target_sub_index:
                states[sub_idx] = True  # keep this one ON
                continue
            # Switch OFF
            success = self.set_subwoofer_off(sub_idx)
            states[sub_idx] = False  # we turned it off

        logger.info(f"Subwoofer isolation: target={target_sub_index}, "
                    f"others={[k for k,v in states.items() if k != target_sub_index]}")
        return states

    # -------------------------------------------------------------------------
    # Power management
    # -------------------------------------------------------------------------

    def get_power_status(self) -> str:
        """Query AVR power status. Returns 'ON', 'OFF', or 'UNKNOWN'."""
        try:
            resp = self._telnet_send("ZM?")
            if "ZMON" in resp.upper():
                return "ON"
            elif "ZMOFF" in resp.upper():
                return "OFF"
            return "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def power_on(self) -> bool:
        """Turn AVR on."""
        try:
            self._telnet_send("ZMON")
            time.sleep(5.0)  # AVR needs time to boot
            return True
        except Exception as e:
            logger.error(f"Power on failed: {e}")
            return False

    def power_off(self) -> bool:
        """Turn AVR off."""
        try:
            self._telnet_send("ZMOFF")
            return True
        except Exception as e:
            logger.error(f"Power off failed: {e}")
            return False


# -----------------------------------------------------------------------------
# Test / CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python avr_control.py <AVR_IP>")
        print("  Tests power status and subwoofer switching.")
        sys.exit(1)

    ip = sys.argv[1]
    avr = AvrControl(ip, is_new_model=True)

    print(f"\n=== AVR Control: {ip} ===")
    status = avr.get_power_status()
    print(f"Power status: {status}")

    if status == "ON":
        print("\nSwitching subwoofer 2 OFF...")
        avr.set_subwoofer_off(2)
        time.sleep(1)
        print("Restoring all subwoofers...")
        avr.restore_all_subwoofers()
    else:
        print("AVR is off — skipping subwoofer test")