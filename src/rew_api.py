"""
REW API Client — speaker-measure
Interfaces with REW (Room EQ Wizard) over HTTP API (localhost:4735).
Used when in "REW Mode" to import recorded sweep responses for deconvolution
and measurement processing via REW instead of local signal_processor.
"""

import http.client
import json
import logging
import socket
import struct
import base64
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

REW_HOST = "127.0.0.1"
REW_PORT = 4735


class RewApiError(Exception):
    """Raised when REW API call fails or REW is not available."""


class RewApiClient:
    """HTTP API client for REW (Room EQ Wizard).

    REW must be running with the API server enabled (-api argument).
    The API is accessible at http://127.0.0.1:4735 by default.

    Usage:
        rew = RewApiClient()
        rew.connect()
        rew.import_impulse_response("/path/to/recorded.wav", measurement_name="FL")
        result = rew.get_measurement_result("FL")
    """

    def __init__(self, host: str = REW_HOST, port: int = REW_PORT):
        self.host = host
        self.port = port
        self._connected = False

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if REW API is responding on localhost."""
        try:
            conn = http.client.HTTPConnection(self.host, self.port, timeout=3.0)
            conn.request("GET", "/")
            resp = conn.getresponse()
            conn.close()
            return resp.status == 200
        except Exception:
            return False

    def connect(self) -> None:
        """Verify REW API is reachable and raise if not."""
        if not self.is_available():
            raise RewApiError(
                f"REW API not available at {self.host}:{self.port}. "
                "Start REW with the -api argument: "
                "C:\\Program Files\\REW\\roomeqwizard.exe -api"
            )
        self._connected = True
        logger.info(f"Connected to REW API at {self.host}:{self.port}")

    # -------------------------------------------------------------------------
    # Generic HTTP helpers
    # -------------------------------------------------------------------------

    def _get(self, path: str) -> dict:
        """GET an endpoint and return parsed JSON."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=10.0)
        conn.request("GET", path)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        if resp.status != 200:
            raise RewApiError(f"GET {path} → HTTP {resp.status}: {data[:200]}")
        return json.loads(data) if data else {}

    def _post(self, path: str, body: Optional[dict] = None, encode_json: bool = True) -> dict:
        """POST to an endpoint with optional JSON body. Returns parsed JSON."""
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30.0)
        headers = {}
        if body and encode_json:
            headers["Content-Type"] = "application/json"
            body_bytes = json.dumps(body).encode("utf-8")
        else:
            body_bytes = b""
        conn.request("POST", path, body=body_bytes, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        if resp.status not in (200, 202):
            raise RewApiError(f"POST {path} → HTTP {resp.status}: {data[:200]}")
        return json.loads(data) if data else {}

    # -------------------------------------------------------------------------
    # Measurement import
    # -------------------------------------------------------------------------

    def import_impulse_response(
        self,
        wav_path: str | Path,
        measurement_name: str = "Imported",
        measurement_notes: str = "",
    ) -> dict:
        """Import a recorded WAV impulse response into REW.

        REW will deconvolve the sweep from the WAV and create a measurement.

        Args:
            wav_path: Path to the recorded WAV file (mic capture of sweep playback).
            measurement_name: Name for this measurement (e.g., "FL", "SW1").
            measurement_notes: Optional notes.

        Returns:
            API response dict with measurement info.
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            raise RewApiError(f"WAV file not found: {wav_path}")

        # Read WAV as raw PCM
        try:
            import soundfile as sf
            audio_data, samplerate = sf.read(str(wav_path), dtype="float32")
        except Exception as e:
            raise RewApiError(f"Failed to read WAV file {wav_path}: {e}")

        # Convert to stereo if mono
        if audio_data.ndim == 1:
            audio_data = np.stack([audio_data, audio_data], axis=1)

        # Build multipart form data manually (REWs API accepts raw PCM in a specific format)
        # REW's import impulse response endpoint accepts:
        # POST /measurements/importimpulseresponse
        # Body: JSON with fields including "data" (base64-encoded float32 PCM)
        # "data" is the raw bytes of the float32 samples, big-endian

        # Pack as big-endian float32 stereo interleaved
        packed = struct.pack(f">{len(audio_data) * 2}f", *audio_data.T.flat)
        b64_data = base64.b64encode(packed).decode("ascii")

        payload = {
            "name": measurement_name,
            "notes": measurement_notes,
            "data": b64_data,
            "sampleRate": int(samplerate),
            "channels": 2,
        }

        logger.info(f"Importing impulse response to REW: {wav_path.name} as '{measurement_name}'")
        result = self._post("/measurements/importimpulseresponse", body=payload)
        logger.info(f"Import result: {result}")
        return result

    def import_measurements_from_folder(
        self,
        folder: str | Path,
        file_to_channel_map: dict[str, str],
    ) -> dict[str, dict]:
        """Import multiple sweep recordings from a folder into REW.

        Args:
            folder: Folder containing recorded WAV files.
            file_to_channel_map: Dict mapping filename (no path) → channel_id (e.g. "FL", "SW1").

        Returns:
            Dict mapping channel_id → API response for each imported measurement.
        """
        folder = Path(folder)
        results = {}

        for filename, channel_id in file_to_channel_map.items():
            wav_path = folder / filename
            if not wav_path.exists():
                logger.warning(f"Sweep file not found, skipping: {wav_path}")
                continue

            try:
                result = self.import_impulse_response(wav_path, measurement_name=channel_id)
                results[channel_id] = result
            except RewApiError as e:
                logger.error(f"Failed to import {filename}: {e}")
                results[channel_id] = {"error": str(e)}

        return results

    # -------------------------------------------------------------------------
    # Measurement results
    # -------------------------------------------------------------------------

    def get_measurements(self) -> list[dict]:
        """Return list of all measurements in REW."""
        return self._get("/measurements")

    def get_measurement(self, measurement_id: int) -> dict:
        """Return details for a specific measurement."""
        return self._get(f"/measurements/{measurement_id}")

    def get_frequency_response(self, measurement_id: int) -> dict:
        """Get frequency response data for a measurement.

        Returns:
            Dict with 'freq' and 'mag' arrays (as base64-encoded float32).
        """
        return self._get(f"/measurements/{measurement_id}/frequencyresponse")

    def delete_measurement(self, measurement_id: int) -> dict:
        """Delete a measurement from REW."""
        return self._post(f"/measurements/{measurement_id}/delete", body={})

    def clear_all_measurements(self) -> dict:
        """Clear all measurements from REW."""
        return self._post("/measurements/clear", body={})

    # -------------------------------------------------------------------------
    # EQ / filters
    # -------------------------------------------------------------------------

    def get_eq_channels(self) -> list[str]:
        """Return list of EQ channel names in REW."""
        return self._get("/eq/channels")

    def get_eq_targets(self, channel: str) -> dict:
        """Get EQ target settings for a channel."""
        return self._get(f"/eq/{channel}/target")

    def set_eq_filters(self, channel: str, filters: list[dict]) -> dict:
        """Apply EQ filters to a channel. See REW API docs for filter format."""
        return self._post(f"/eq/{channel}/filters", body={"filters": filters})