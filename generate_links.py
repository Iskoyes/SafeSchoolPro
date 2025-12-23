import os, json, secrets, string, csv
from pathlib import Path
from dotenv import load_dotenv
import io, csv

try:
    import qrcode
    QR_OK = True
except Exception:
    QR_OK = False

load_dotenv()
BOT_USERNAME = os.getenv("BOT_USERNAME")  
if not BOT_USERNAME:
    BOT_USERNAME = input("Enter your bot username (without @): ").strip()

TOKENS_FILE = Path("pending_tokens.json")  
QR_DIR = Path("qr_links")
QR_DIR.mkdir(exist_ok=True)

def gen_token(prefix="bind", n=8):
    alphabet = string.ascii_lowercase + string.digits
    return f"{prefix}-" + "".join(secrets.choice(alphabet) for _ in range(n))

def load_tokens():
    if TOKENS_FILE.exists():
        return json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
    return {}

def save_tokens(tokens: dict):
    TOKENS_FILE.write_text(json.dumps(tokens, ensure_ascii=False, indent=2), encoding="utf-8")

def make_link(token: str) -> str:
    return f"https://t.me/{BOT_USERNAME}?start={token}"


students = [
    # "ivan_petrov",
    # "aidos_k",
]

with io.open("students.csv", "r", encoding="utf-8-sig") as f:
    for row in csv.reader(f):
        if not row: continue
        sid = row[0].lstrip("\ufeff").strip()
        if sid.lower() in ("student_id","id",""): continue
        students.append(sid)
if not students:
    print("No students provided. Add students to 'students.csv' or hardcode in the list.")
    raise SystemExit


pending = load_tokens()
links = {} 

for s in students:

    existing = next((t for t, sid in pending.items() if sid == s), None)
    token = existing or gen_token()
    pending[token] = s
    links[s] = make_link(token)


    if QR_OK:
        img = qrcode.make(links[s])
        img.save(QR_DIR / f"qr_{s}.png")


save_tokens(pending)


with open("links.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["student_id", "deep_link"])
    for s, link in links.items():
        w.writerow([s, link])

print("Done.")
print(f"Saved tokens -> {TOKENS_FILE}")
print("Links:")
for s, link in links.items():
    print(f"  {s}: {link}")
print(f"QR images (if qrcode installed) -> {QR_DIR}/qr_<student_id>.png")
print("Also wrote links.csv for easy sharing.")
