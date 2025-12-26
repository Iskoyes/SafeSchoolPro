import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import os
import requests
from insightface.app import FaceAnalysis
from urllib.parse import quote_plus

from recognition.settings import RecognitionSettings, SettingsStore

# Constants preserved from the legacy pipeline
STABLE_SEC = 2.0
EVENT_COOLDOWN_SEC = 120
SMOOTH_ALPHA = 0.65
PROPAGATE_MAX_AGE = 2.0


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def cosine_sim_matrix(A, b):
    return A @ b


def load_db():
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        base_dir / "faces_db.npz",
        base_dir / "safeschool-recognition" / "data" / "faces_db.npz",
        base_dir / "safeschool-recognition" / "faces_db.npz",
    ]
    for p in candidates:
        if p.exists():
            db = np.load(p, allow_pickle=True)
            names = list(db["names"])
            embs = db["embs"]
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.clip(norms, 1e-9, None)
            return names, embs
    raise FileNotFoundError("faces_db.npz не найден в ожидаемых путях: " + ", ".join(map(str, candidates)))


def load_parents():
    base_dir = Path(__file__).resolve().parent.parent
    candidates = [
        base_dir / "parents.json",
        base_dir / "safeschool-recognition" / "data" / "parents.json",
        base_dir / "safeschool-recognition" / "parents.json",
    ]
    for p in candidates:
        if p.exists():
            return p, p.read_text(encoding="utf-8")
    return None, "{}"


@dataclass
class RecognitionEvent:
    name: str
    timestamp: str
    similarity: float
    text: str


@dataclass
class TelegramStatus:
    timestamp: str
    message: str
    success: bool


class TelegramNotifier:
    def __init__(self, token: Optional[str], parents: Dict[str, List[str]], status_callback=None):
        self.token = token
        self.parents = parents
        self.status_history: deque[TelegramStatus] = deque(maxlen=50)
        self.status_callback = status_callback

    def _record(self, status: TelegramStatus):
        self.status_history.append(status)
        if self.status_callback:
            self.status_callback({"type": "status", "data": asdict(status), "source": "telegram"})

    def notify(self, student_id: str, text: str, frame=None):
        if not self.token:
            self._record(TelegramStatus(timestamp=now_str(), message="Telegram token отсутствует", success=False))
            return

        chat_ids = self.parents.get(student_id, [])
        if not chat_ids:
            self._record(
                TelegramStatus(
                    timestamp=now_str(),
                    message=f"Нет родителей для {student_id}, уведомление пропущено",
                    success=False,
                )
            )
            return

        def _send():
            for cid in chat_ids:
                try:
                    requests.get(
                        f"https://api.telegram.org/bot{self.token}/sendMessage",
                        params={"chat_id": cid, "text": text, "disable_notification": "true"},
                        timeout=5,
                    )
                    if frame is not None:
                        _, buf = cv2.imencode(".jpg", frame)
                        requests.post(
                            f"https://api.telegram.org/bot{self.token}/sendPhoto",
                            data={"chat_id": cid},
                            files={"photo": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                            timeout=5,
                        )
                    self._record(
                        TelegramStatus(
                            timestamp=now_str(), message=f"Отправлено в чат {cid}: {text}", success=True
                        )
                    )
                except Exception as e:
                    self._record(
                        TelegramStatus(
                            timestamp=now_str(),
                            message=f"Ошибка отправки в чат {cid}: {e}",
                            success=False,
                        )
                    )

        threading.Thread(target=_send, daemon=True).start()


class RecognitionStream:
    """
    Runs face recognition in a background thread and exposes frames + events via thread-safe queues.
    """

    def __init__(self, settings_store: Optional[SettingsStore] = None):
        self.settings_store = settings_store or SettingsStore()
        self.settings = self.settings_store.load()

        self.names, self.embs = load_db()
        _, parents_content = load_parents()
        try:
            import json

            self.parents = json.loads(parents_content)
        except Exception:
            self.parents = {}

        token = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TOKEN") or "").strip()
        self.notifier = TelegramNotifier(token, self.parents, status_callback=self._push_message)

        self.face_app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0, det_size=self.settings.detection_size)

        self._stop_event = threading.Event()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._frame_queue: queue.Queue[bytes] = queue.Queue(maxsize=5)
        self._message_queue: queue.Queue[dict] = queue.Queue()
        self._recent_events: deque[RecognitionEvent] = deque(maxlen=50)
        self._last_error: Optional[str] = None

        self._seen_info: Dict[str, Dict] = {}
        self._last_event_at: Dict[str, float] = {}
        self._frame_idx = 0
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------- public API -------------
    def start(self):
        if self._running.is_set():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._push_message(
            {
                "type": "status",
                "data": {"timestamp": now_str(), "message": "Stream stopped", "running": False},
                "source": "runtime",
            }
        )

    def update_settings(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self.settings_store.save(self.settings)

    def next_frame(self, timeout: float = 1.0) -> Optional[bytes]:
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def frames(self) -> Iterable[bytes]:
        while True:
            frame = self.next_frame(timeout=1.0)
            if frame is None:
                if not self._running.is_set():
                    break
                continue
            yield frame

    def messages(self, timeout: float = 1.0) -> Optional[dict]:
        try:
            return self._message_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def recent_events(self) -> List[RecognitionEvent]:
        return list(self._recent_events)

    def status(self) -> dict:
        return {
            "running": self._running.is_set(),
            "last_error": self._last_error,
            "settings": asdict(self.settings),
            "recent_events": [asdict(e) for e in self._recent_events],
            "telegram": [asdict(s) for s in self.notifier.status_history],
        }

    # ------------- internal logic -------------
    def _push_message(self, payload: dict):
        try:
            self._message_queue.put_nowait(payload)
        except queue.Full:
            pass

    def _record_event(self, event: RecognitionEvent):
        self._recent_events.append(event)
        self._push_message({"type": "event", "data": asdict(event)})

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        candidates = []
        if self.settings.camera_source:
            candidates.append(self.settings.camera_source)
            if isinstance(self.settings.camera_source, str) and "rtsp" not in self.settings.camera_source:
                ip_only = self.settings.camera_source.split("://")[-1].split("/", 1)[0]
                candidates.append(f"rtsp://{ip_only}:554/")

        # Legacy fallback using env to try RTSP port 554 when only IP is provided
        ip_url = os.getenv("IP_CAMERA_URL", "").strip()
        ip_user = os.getenv("IP_CAMERA_USER", "").strip()
        ip_pass = os.getenv("IP_CAMERA_PASS", "").strip()
        if ip_url and "rtsp" not in ip_url:
            ip_only = ip_url.split("://")[-1].split("/", 1)[0]
            if ip_user and ip_pass:
                candidates.append(f"rtsp://{quote_plus(ip_user)}:{quote_plus(ip_pass)}@{ip_only}:554/")
            else:
                candidates.append(f"rtsp://{ip_only}:554/")
        candidates.append(0)

        for src in candidates:
            cap = None
            try:
                if isinstance(src, str) and src.startswith("rtsp"):
                    cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                else:
                    cap = cv2.VideoCapture(src)
                time.sleep(0.6)
                if cap.isOpened():
                    w, h = self.settings.desired_resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                    return cap
                cap.release()
            except Exception as e:
                self._last_error = f"Ошибка открытия {src}: {e}"
        self._last_error = "Не удалось открыть камеру"
        return None

    def _run(self):
        self._cap = self._open_capture()
        if not self._cap:
            self._running.clear()
            self._push_message(
                {
                    "type": "status",
                    "data": {"timestamp": now_str(), "message": self._last_error, "running": False},
                    "source": "runtime",
                }
            )
            return

        self._running.set()
        self._push_message(
            {
                "type": "status",
                "data": {"timestamp": now_str(), "message": "Stream started", "running": True},
                "source": "runtime",
            }
        )
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok:
                self._last_error = "Камера недоступна."
                break

            self._frame_idx += 1
            display_frame = frame.copy()
            process_this = self._frame_idx % max(1, self.settings.process_every_n) == 0
            faces = []

            now_ts = time.time()
            if process_this:
                small = cv2.resize(frame, (0, 0), fx=self.settings.process_scale, fy=self.settings.process_scale)
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                faces = self.face_app.get(rgb)

            detected_names = set()
            for f in faces:
                scale_inv = 1.0 / self.settings.process_scale
                x1, y1, x2, y2 = (f.bbox * scale_inv).astype(int)
                emb = f.embedding
                emb = emb / np.linalg.norm(emb)
                sims = cosine_sim_matrix(self.embs, emb)
                idx = int(np.argmax(sims))
                sim_best = float(sims[idx])
                name = self.names[idx] if sim_best >= self.settings.sim_threshold else None

                color = (0, 255, 0) if name else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name or 'unknown'} ({sim_best:.2f})"
                cv2.putText(
                    display_frame,
                    label,
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                if name:
                    detected_names.add(name)
                    info = self._seen_info.get(name, {"stable_since": None, "last_seen": 0})
                    max_gap = 1.0
                    if info["last_seen"] == 0 or (now_ts - info["last_seen"]) > max_gap:
                        info["stable_since"] = now_ts
                    info["last_seen"] = now_ts
                    info["target_bbox"] = (x1, y1, x2, y2)
                    info["bbox"] = (x1, y1, x2, y2)
                    info["sim"] = sim_best
                    info["label"] = label
                    info.setdefault("smooth_bbox", info["bbox"])
                    self._seen_info[name] = info

                    held = now_ts - (info["stable_since"] or now_ts)
                    cv2.putText(
                        display_frame,
                        f"stable: {held:.1f}s/{STABLE_SEC}s",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                    if held >= STABLE_SEC:
                        last_t = self._last_event_at.get(name, 0)
                        if now_ts - last_t >= EVENT_COOLDOWN_SEC:
                            text = f"✅ {name} вошёл в школу в {now_str()}"
                            tframe = display_frame.copy()
                            self.notifier.notify(name, text, tframe)
                            event = RecognitionEvent(
                                name=name,
                                timestamp=now_str(),
                                similarity=sim_best,
                                text=text,
                            )
                            self._record_event(event)
                            self._last_event_at[name] = now_ts
                            info["stable_since"] = None
                            self._seen_info[name] = info

            current_display_names = []
            for n, info in list(self._seen_info.items()):
                age = now_ts - info.get("last_seen", 0)
                if age > PROPAGATE_MAX_AGE:
                    del self._seen_info[n]
                    continue

                target = info.get("target_bbox", info.get("bbox"))
                if target is None:
                    continue

                sx1, sy1, sx2, sy2 = info.get("smooth_bbox", target)
                tx1, ty1, tx2, ty2 = target
                sx1 = int(SMOOTH_ALPHA * tx1 + (1 - SMOOTH_ALPHA) * sx1)
                sy1 = int(SMOOTH_ALPHA * ty1 + (1 - SMOOTH_ALPHA) * sy1)
                sx2 = int(SMOOTH_ALPHA * tx2 + (1 - SMOOTH_ALPHA) * sx2)
                sy2 = int(SMOOTH_ALPHA * ty2 + (1 - SMOOTH_ALPHA) * sy2)
                info["smooth_bbox"] = (sx1, sy1, sx2, sy2)
                info["bbox"] = target
                self._seen_info[n] = info

                color = (0, 255, 0)
                cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), color, 2)
                label_text = info.get("label", n)
                cv2.putText(
                    display_frame,
                    label_text,
                    (sx1, max(20, sy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

                if info.get("stable_since"):
                    held = now_ts - (info["stable_since"] or now_ts)
                    cv2.putText(
                        display_frame,
                        f"stable: {held:.1f}s/{STABLE_SEC}s",
                        (sx1, sy2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                current_display_names.append(n)

            if current_display_names:
                names_str = ", ".join(sorted(set(current_display_names)))
                cv2.putText(
                    display_frame,
                    f"detected: {names_str}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            ok, buf = cv2.imencode(".jpg", display_frame)
            if ok:
                try:
                    self._frame_queue.put_nowait(buf.tobytes())
                except queue.Full:
                    try:
                        _ = self._frame_queue.get_nowait()
                        self._frame_queue.put_nowait(buf.tobytes())
                    except queue.Empty:
                        pass

        self._running.clear()
        self._push_message(
            {
                "type": "status",
                "data": {"timestamp": now_str(), "message": self._last_error, "running": False},
                "source": "runtime",
            }
        )
