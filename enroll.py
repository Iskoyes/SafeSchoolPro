import cv2, time, numpy as np
from insightface.app import FaceAnalysis
from pathlib import Path


AUTO_MODE = True          # True = автоматический захват после CAPTURE_AFTER сек удержания
CAPTURE_AFTER = 2.0
FRAMES_TO_CAPTURE = 15

NAME = input("Введите уникальный ID ученика (латиницей, напр. ivan_petrov): ").strip()

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

cap = cv2.VideoCapture(0)
cv2.namedWindow("Enroll - наведите лицо на камеру", cv2.WINDOW_NORMAL)

def pick_largest_face(faces):
    if not faces: return None
    areas = [(f, (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])) for f in faces]
    return max(areas, key=lambda x: x[1])[0]

def capture_embeddings(n=FRAMES_TO_CAPTURE):
    embeds = []
    count = 0
    while count < n:
        ok, fr = cap.read()
        if not ok: break
        fs = app.get(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        f = pick_largest_face(fs)
        if f is not None:
            emb = f.embedding
            emb = emb / np.linalg.norm(emb)
            embeds.append(emb)
            count += 1
            cv2.putText(fr, f"captured {count}/{n}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow("Enroll - захват", fr)
        if (cv2.waitKey(50) & 0xFF) in (ord('q'), 27):
            break
    return embeds

embeds = []
stable_since = None

print("Режим:", "АВТО (без кнопок)" if AUTO_MODE else "КЛАССИЧЕСКИЙ (c/q)")
if not AUTO_MODE:
    print("Нажмите 'c' чтобы начать захват, 'q' — выход.")

while True:
    ok, frame = cap.read()
    if not ok: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(rgb)
    f = pick_largest_face(faces)

    show = frame.copy()
    if f is not None:
        x1,y1,x2,y2 = f.bbox.astype(int)
        cv2.rectangle(show,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("Enroll - наведите лицо на камеру", show)
    key = cv2.waitKey(10) & 0xFF

    if AUTO_MODE:
        if f is not None:
            if stable_since is None:
                stable_since = time.time()
            held = time.time() - stable_since
            cv2.putText(show, f"hold: {held:.1f}s / {CAPTURE_AFTER}s", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow("Enroll - наведите лицо на камеру", show)
            if held >= CAPTURE_AFTER:
                print("Стабильное лицо найдено. Начинаю захват...")
                embeds = capture_embeddings(FRAMES_TO_CAPTURE)
                break
        else:
            stable_since = None

        if key in (ord('q'), 27):
            break

    else:
        if key == ord('c'):
            print("Захват кадров...")
            embeds = capture_embeddings(FRAMES_TO_CAPTURE)
            print("Готово.")
        elif key in (ord('q'), 27):
            break

cap.release(); cv2.destroyAllWindows()

if len(embeds) == 0:
    print("Не удалось снять эмбеддинги. Повторите.")
    raise SystemExit

avg_emb = np.mean(np.stack(embeds), axis=0)
avg_emb = avg_emb / np.linalg.norm(avg_emb)

# save to faces_db.npz
try:
    db = np.load("faces_db.npz", allow_pickle=True)
    names = list(db["names"])
    embs  = list(db["embs"])
except FileNotFoundError:
    names, embs = [], []

if NAME in names:
    idx = names.index(NAME)
    embs[idx] = avg_emb
else:
    names.append(NAME); embs.append(avg_emb)

np.savez_compressed("faces_db.npz", names=np.array(names, dtype=object), embs=np.stack(embs))
print(f"Сохранено: {NAME}. Всего учеников в БД: {len(names)}")

# --- интеграция: создать одноразовую ссылку и QR для родителя ---
try:
    from pending_tokens import create_bind_link_for_student
    res = create_bind_link_for_student(NAME)
    print("Deep link для привязки родителя:", res["link"])
    if res.get("qr"):
        print("QR сохранён в:", res["qr"])
    else:
        print("QR не создан (установите библиотеку 'qrcode' чтобы генерировать QR).")
except Exception as e:
    print("Не удалось создать deep link:", e)

# end enroll.py
