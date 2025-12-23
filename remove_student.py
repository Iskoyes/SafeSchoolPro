import sys
import json
import csv
import shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
FB = ROOT / "faces_db.npz"
PARENTS = ROOT / "parents.json"
LINKS = ROOT / "links.csv"
PENDING = ROOT / "pending_tokens.json"
QR_DIR = ROOT / "qr_links"

def backup(p: Path):
    if p.exists():
        shutil.copy(str(p), str(p) + ".bak")

def remove_from_faces_db(student_id: str):
    if not FB.exists():
        print("faces_db.npz не найден, пропускаю.")
        return
    backup(FB)
    db = np.load(FB, allow_pickle=True)
    names = list(db["names"])
    embs = list(db["embs"])
    if student_id not in names:
        print("Студент не найден в faces_db.npz.")
    else:
        idx = names.index(student_id)
        del names[idx]
        del embs[idx]
        if len(names) == 0:
            FB.unlink()
            print("Удалил faces_db.npz — база пуста.")
        else:
            np.savez_compressed(FB, names=np.array(names, dtype=object), embs=np.stack(embs))
            print(f"Удалил {student_id} из faces_db.npz")

def remove_from_parents(student_id: str):
    if not PARENTS.exists():
        print("parents.json не найден, пропускаю.")
        return
    backup(PARENTS)
    d = json.loads(PARENTS.read_text(encoding="utf-8"))
    if student_id in d:
        del d[student_id]
        if d:
            PARENTS.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            PARENTS.unlink()
        print(f"Удалил {student_id} из parents.json")
    else:
        print("Студент не найден в parents.json.")

def remove_from_links(student_id: str):
    if not LINKS.exists():
        print("links.csv не найден, пропускаю.")
        return
    backup(LINKS)
    rows = []
    removed = False
    with LINKS.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if r.get("student_id") == student_id:
                removed = True
                continue
            rows.append(r)
    if removed:
        with LINKS.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["student_id","deep_link"])
            for r in rows:
                w.writerow([r["student_id"], r["deep_link"]])
        print(f"Удалил {student_id} из links.csv")
    else:
        print("Студент не найден в links.csv.")

def remove_from_pending(student_id: str):
    if not PENDING.exists():
        print("pending_tokens.json не найден, пропускаю.")
        return
    backup(PENDING)
    d = json.loads(PENDING.read_text(encoding="utf-8"))
    keys = [k for k,v in d.items() if v == student_id]
    if not keys:
        print("Студент не найден в pending_tokens.json.")
        return
    for k in keys:
        del d[k]
    if d:
        PENDING.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        PENDING.unlink()
    print(f"Удалил токены для {student_id} из pending_tokens.json: {keys}")

def remove_qr(student_id: str):
    p = QR_DIR / f"{student_id}.png"
    if p.exists():
        p.unlink()
        print(f"Удалён QR: {p}")
    else:
        print("QR-файл не найден, пропускаю.")

def main():
    if len(sys.argv) < 2:
        print("Usage: py remove_student.py <student_id>")
        raise SystemExit(1)
    sid = sys.argv[1].strip()
    print("Backup файлов будет создан (если они есть). Удаляю:", sid)
    remove_from_faces_db(sid)
    remove_from_parents(sid)
    remove_from_links(sid)
    remove_from_pending(sid)
    remove_qr(sid)
    print("Готово. Проверьте файлы: faces_db.npz, parents.json, links.csv, pending_tokens.json, qr_links/")

if __name__ == "__main__":
    main()