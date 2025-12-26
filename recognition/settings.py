import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load .env from current dir first, then try safeschool-recognition/.env
if not load_dotenv():
    alt = Path(__file__).resolve().parent.parent / "safeschool-recognition" / ".env"
    if alt.exists():
        load_dotenv(str(alt))


DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "settings.json"


def _default_camera_source() -> str:
    """
    Recreates the legacy camera source resolution logic with IP credentials support.
    """
    ip_url = os.getenv("IP_CAMERA_URL", "").strip()
    ip_user = os.getenv("IP_CAMERA_USER", "").strip()
    ip_pass = os.getenv("IP_CAMERA_PASS", "").strip()

    if ip_url:
        if ip_user and ip_pass and "@" not in ip_url:
            url = ip_url
            if "://" not in url:
                url = "rtsp://" + url
            scheme, rest = url.split("://", 1)
            creds = f"{quote_plus(ip_user)}:{quote_plus(ip_pass)}@"
            return f"{scheme}://{creds}{rest}"
        return ip_url
    return os.getenv("CAMERA_SOURCE", "").strip()


@dataclass
class RecognitionSettings:
    camera_source: str = _default_camera_source() or ""
    process_every_n: int = int(os.getenv("PROCESS_EVERY_N", "2"))
    sim_threshold: float = float(os.getenv("SIM_THRESHOLD", "0.38"))
    process_scale: float = float(os.getenv("PROCESS_SCALE", "0.6"))
    detection_size: Tuple[int, int] = (320, 320)
    desired_resolution: Tuple[int, int] = (
        int(os.getenv("CAM_WIDTH", "1280")),
        int(os.getenv("CAM_HEIGHT", "720")),
    )


class SettingsStore:
    """
    A tiny helper to persist runtime-tunable recognition parameters between runs.
    """

    def __init__(self, path: Path = DEFAULT_SETTINGS_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> RecognitionSettings:
        if not self.path.exists():
            return RecognitionSettings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return RecognitionSettings(
                camera_source=data.get("camera_source", RecognitionSettings.camera_source),
                process_every_n=int(data.get("process_every_n", RecognitionSettings.process_every_n)),
                sim_threshold=float(data.get("sim_threshold", RecognitionSettings.sim_threshold)),
                process_scale=float(data.get("process_scale", RecognitionSettings.process_scale)),
                detection_size=tuple(data.get("detection_size", RecognitionSettings.detection_size)),
                desired_resolution=tuple(
                    data.get("desired_resolution", RecognitionSettings.desired_resolution)
                ),
            )
        except Exception:
            # Fallback to defaults when parsing fails
            return RecognitionSettings()

    def save(self, settings: RecognitionSettings) -> None:
        payload = asdict(settings)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
