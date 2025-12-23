import os, time, cv2, json, numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis
from dotenv import load_dotenv
from urllib.parse import quote_plus
import threading  # добавлено

# --- config ---
SIM_THRESHOLD = 0.38
STABLE_SEC = 2.0
EVENT_COOLDOWN_SEC = 120
CAM_INDEX = 0

# Rate / scale tuning to speed up processing
PROCESS_EVERY_N = int(os.getenv("PROCESS_EVERY_N", "2"))    # обработка каждого N-го кадра (увеличьте для ещё большей скорости)
PROCESS_SCALE = float(os.getenv("PROCESS_SCALE", "0.6"))    # масштаб входного кадра для детекции/эмбеддинга
DETECTION_SIZE = (320, 320)   # размер детектора/анализа для insightface (уменьшение ускоряет работу)

# try load .env from current dir, otherwise from safeschool-recognition/.env
loaded = load_dotenv()
if not loaded:
    alt = os.path.join(os.path.dirname(__file__), "safeschool-recognition", ".env")
    if os.path.exists(alt):
        load_dotenv(alt)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TOKEN")

IP_CAMERA_URL = os.getenv("IP_CAMERA_URL", "").strip()
IP_CAMERA_USER = os.getenv("IP_CAMERA_USER", "").strip()
IP_CAMERA_PASS = os.getenv("IP_CAMERA_PASS", "").strip()

# если в .env указан полный RTSP (с логином) — используем его как есть
# иначе, если заданы user/pass + IP, подставим их (экранируем)
if IP_CAMERA_URL:
    if IP_CAMERA_USER and IP_CAMERA_PASS and "@" not in IP_CAMERA_URL:
        u = IP_CAMERA_URL
        if "://" not in u:
            u = "rtsp://" + u
        scheme, rest = u.split("://", 1)
        creds = f"{quote_plus(IP_CAMERA_USER)}:{quote_plus(IP_CAMERA_PASS)}@"
        IP_CAMERA_SOURCE = f"{scheme}://{creds}{rest}"
    else:
        IP_CAMERA_SOURCE = IP_CAMERA_URL
else:
    IP_CAMERA_SOURCE = ""

# приоритет — IP поток, иначе локальная камера
print("IP_CAMERA_SOURCE =", IP_CAMERA_SOURCE)
candidates = []
if IP_CAMERA_SOURCE:
    candidates.append(IP_CAMERA_SOURCE)
# добавим возможный RTSP-адрес с портом 554 на случай, если в .env указан просто IP
if IP_CAMERA_URL and "rtsp" not in IP_CAMERA_URL:
    ip_only = IP_CAMERA_URL.split("://")[-1].split("/", 1)[0]
    if IP_CAMERA_USER and IP_CAMERA_PASS:
        candidates.append(f"rtsp://{quote_plus(IP_CAMERA_USER)}:{quote_plus(IP_CAMERA_PASS)}@{ip_only}:554/")
# локальная камера в конце
candidates.append(CAM_INDEX)

def try_open(src, backend=None, wait=0.6):
    try:
        cap = cv2.VideoCapture(src) if backend is None else cv2.VideoCapture(src, backend)
        time.sleep(wait)
        ok = cap.isOpened()
        print(f"try_open: src={src!r} backend={backend} opened={ok}")
        if ok:
            return cap
        cap.release()
    except Exception as e:
        print("Ошибка открытия", src, ":", e)
    return None

cap = None
for src in candidates:
    # для RTSP лучше сначала попробовать с FFMPEG backend (более стабильный)
    if isinstance(src, str) and src.startswith("rtsp"):
        cap = try_open(src, cv2.CAP_FFMPEG)
        if cap:
            break
    cap = try_open(src)
    if cap:
        break

if cap is None:
    print("Не удалось открыть IP-камеру. Открываю локальную камеру (index 0) в качестве fallback.")
    cap = cv2.VideoCapture(CAM_INDEX)

# Попытка задать разрешение и масштабировать окно под реальное кадр
DESIRED_W = int(os.getenv("CAM_WIDTH", "1280"))
DESIRED_H = int(os.getenv("CAM_HEIGHT", "720"))
try:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, DESIRED_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DESIRED_H)
    # небольшой тайм-аут чтобы поток успел переинициализироваться
    time.sleep(0.5)
except Exception as e:
    print("Не удалось задать разрешение:", e)

WIN_NAME = "Safeschool - Recognition"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

# Считаем один кадр чтобы узнать реальный размер и подогнать окно
_ok, _tmp = cap.read()
if _ok and _tmp is not None:
    h, w = _tmp.shape[:2]
    try:
        cv2.resizeWindow(WIN_NAME, w, h)
        print(f"Окно подогнано под разрешение потока: {w}x{h}")
    except Exception:
        # иногда resizeWindow игнорируется (платформенные ограничения) — просто игнорируем
        pass
else:
    print("Не удалось получить тестовый кадр для определения разрешения потока.")

# --- helper functions / loading DB / notifier ---
def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def cosine_sim_matrix(A, b):
    return A @ b

def load_db():
    # try several locations for faces_db.npz
    candidates = [
        os.path.join(os.path.dirname(__file__), "faces_db.npz"),
        os.path.join(os.path.dirname(__file__), "safeschool-recognition", "data", "faces_db.npz"),
        os.path.join(os.path.dirname(__file__), "safeschool-recognition", "faces_db.npz"),
    ]
    for p in candidates:
        if os.path.exists(p):
            db = np.load(p, allow_pickle=True)
            names = list(db["names"])
            embs = db["embs"]
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / np.clip(norms, 1e-9, None)
            print("Загружена БД лиц:", p)
            return names, embs
    raise FileNotFoundError("faces_db.npz не найден в ожидаемых путях: " + ", ".join(candidates))

def load_parents():
    candidates = [
        os.path.join(os.path.dirname(__file__), "parents.json"),
        os.path.join(os.path.dirname(__file__), "safeschool-recognition", "data", "parents.json"),
        os.path.join(os.path.dirname(__file__), "safeschool-recognition", "parents.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                print("Загружен parents.json:", p)
                return json.load(f)
    print("parents.json не найден, продолжу с пустым списком родителей.")
    return {}

def notify_parents(student_id: str, text: str, frame=None):
    if not TOKEN:
        print("TOKEN не настроен, уведомление не отправлено:", text)
        return
    chat_ids = PARENTS.get(student_id, [])
    for cid in chat_ids:
        try:
            requests.get(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                params={"chat_id": cid, "text": text, "disable_notification": "true"},
                timeout=5,
            )
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame)
                requests.post(
                    f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                    data={"chat_id": cid},
                    files={"photo": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                    timeout=5,
                )
        except Exception as e:
            print("Telegram error:", e)

# lazy import requests (used only in notify_parents)
import requests

# load data and init face analyzer
NAMES, EMBS = load_db()
PARENTS = load_parents()

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

# state for multi-face events
seen_info = {}
last_event_at = {}

frame_idx = 0  # добавлено

print("Готово. Держите лицо в кадре 2–3 сек для срабатывания события. Выход: Q/ESC.")
while True:
    ok, frame = cap.read()
    if not ok:
        print("Камера недоступна.")
        break

    frame_idx += 1
    # показываем живой фрейм, но распознаём не каждый кадр и на уменьшенном изображении
    process_this = (frame_idx % PROCESS_EVERY_N == 0)

    display_frame = frame.copy()

    if process_this:
        # уменьшение для ускорения детекции/эмбеддинга
        small = cv2.resize(frame, (0, 0), fx=PROCESS_SCALE, fy=PROCESS_SCALE)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)
    else:
        faces = []  # пропускаем распознавание на этом кадре

    detected_names = set()
    now = time.time()

    for f in faces:
        # масштабируем bbox и координаты обратно на исходный размер
        scale_inv = 1.0 / PROCESS_SCALE
        x1, y1, x2, y2 = (f.bbox * scale_inv).astype(int)
        emb = f.embedding
        emb = emb / np.linalg.norm(emb)
        sims = cosine_sim_matrix(EMBS, emb)
        idx = int(np.argmax(sims))
        sim_best = float(sims[idx])
        name = NAMES[idx] if sim_best >= SIM_THRESHOLD else None

        color = (0, 255, 0) if name else (0, 0, 255)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name or 'unknown'} ({sim_best:.2f})"
        cv2.putText(display_frame, label, (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if name:
            detected_names.add(name)

            info = seen_info.get(name, {"stable_since": None, "last_seen": 0})
            MAX_GAP = 1.0
            if info["last_seen"] == 0 or (now - info["last_seen"]) > MAX_GAP:
                info["stable_since"] = now
            info["last_seen"] = now
            # сохраняем цельный bbox и метки; инициализируем smooth если нужно
            info["target_bbox"] = (x1, y1, x2, y2)
            info["bbox"] = (x1, y1, x2, y2)
            info["sim"] = sim_best
            info["label"] = label
            if "smooth_bbox" not in info:
                info["smooth_bbox"] = info["bbox"]
            seen_info[name] = info

            held = now - (info["stable_since"] or now)
            cv2.putText(display_frame, f"stable: {held:.1f}s/{STABLE_SEC}s", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if held >= STABLE_SEC:
                last_t = last_event_at.get(name, 0)
                if now - last_t >= EVENT_COOLDOWN_SEC:
                    text = f"✅ {name} вошёл в школу в {now_str()}"
                    tframe = display_frame.copy()
                    threading.Thread(target=notify_parents, args=(name, text, tframe), daemon=True).start()
                    print(text)
                    last_event_at[name] = now
                    info["stable_since"] = None
                    seen_info[name] = info

    # обновляем сглаженные bbox и рисуем ВСЕ метки каждый кадр (чтобы не было мерцания)
    SMOOTH_ALPHA = float(os.getenv("SMOOTH_ALPHA", "0.65"))  # 0..1, ближе к 1 быстрее реагирует
    PROPAGATE_MAX_AGE = float(os.getenv("PROPAGATE_MAX_AGE", "2.0"))  # сколько секунд рисовать без новых детекций

    # обновление smooth_bbox и рисование по всем недавно виденным людям
    current_display_names = []
    for n, info in list(seen_info.items()):
        age = now - info.get("last_seen", 0)
        if age > PROPAGATE_MAX_AGE:
            # удаляем устаревшие записи
            del seen_info[n]
            continue

        # вычисляем цельный bbox (если была новая детекция, target_bbox есть)
        target = info.get("target_bbox", info.get("bbox"))
        if target is None:
            continue

        # smooth update
        sx1, sy1, sx2, sy2 = info.get("smooth_bbox", target)
        tx1, ty1, tx2, ty2 = target
        sx1 = int(SMOOTH_ALPHA * tx1 + (1 - SMOOTH_ALPHA) * sx1)
        sy1 = int(SMOOTH_ALPHA * ty1 + (1 - SMOOTH_ALPHA) * sy1)
        sx2 = int(SMOOTH_ALPHA * tx2 + (1 - SMOOTH_ALPHA) * sx2)
        sy2 = int(SMOOTH_ALPHA * ty2 + (1 - SMOOTH_ALPHA) * sy2)
        info["smooth_bbox"] = (sx1, sy1, sx2, sy2)
        # обновим bbox для совместимости
        info["bbox"] = target
        seen_info[n] = info

        # рисуем рамку и метку на текущем кадре (всегда)
        color = (0, 255, 0)
        cv2.rectangle(display_frame, (sx1, sy1), (sx2, sy2), color, 2)
        label_text = info.get("label", n)
        cv2.putText(display_frame, label_text, (sx1, max(20, sy1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # рисуем стабильность, если есть
        if info.get("stable_since"):
            held = now - (info["stable_since"] or now)
            cv2.putText(display_frame, f"stable: {held:.1f}s/{STABLE_SEC}s", (sx1, sy2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        current_display_names.append(n)

    # показываем строку с именами на каждом кадре, на основе seen_info (а не только process_this)
    if current_display_names:
        names_str = ", ".join(sorted(set(current_display_names)))
        cv2.putText(display_frame, f"detected: {names_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Safeschool - Recognition", display_frame)
    key = cv2.waitKey(10) & 0xFF
    if key in (ord('q'), ord('Q'), 27):
        break

cap.release()
cv2.destroyAllWindows()
