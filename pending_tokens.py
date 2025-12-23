import os
import json
import csv
import secrets
import string
from pathlib import Path
from dotenv import load_dotenv

# load .env from cwd, fallback to safeschool-recognition/.env
load_dotenv()
if not os.getenv("BOT_USERNAME"):
    alt = Path(__file__).parent / "safeschool-recognition" / ".env"
    if alt.exists():
        load_dotenv(str(alt))

ROOT = Path(__file__).parent
PENDING_F = ROOT / "pending_tokens.json"
LINKS_F = ROOT / "links.csv"
QR_DIR = ROOT / "qr_links"
QR_DIR.mkdir(exist_ok=True)

def gen_token(prefix="bind", n=8):
    alphabet = string.ascii_lowercase + string.digits
    return f"{prefix}-" + "".join(secrets.choice(alphabet) for _ in range(n))

def load_pending():
    if PENDING_F.exists():
        try:
            return json.loads(PENDING_F.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_pending(d: dict):
    PENDING_F.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

def load_links():
    links = {}
    if LINKS_F.exists():
        try:
            with LINKS_F.open("r", encoding="utf-8", newline="") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    links[r["student_id"]] = r["deep_link"]
        except Exception:
            pass
    return links

def save_links(links: dict):
    with LINKS_F.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "deep_link"])
        for sid, link in links.items():
            w.writerow([sid, link])

def make_link(bot_username: str, token: str) -> str:
    return f"https://t.me/{bot_username}?start={token}"

def _make_qr_image(link: str, out_path: Path):
    try:
        import qrcode
        img = qrcode.make(link)
        img.save(str(out_path))
        return True
    except Exception:
        # qrcode not available — skip QR creation
        return False

def create_bind_link_for_student(student_id: str, bot_username: str = None, reuse=True):
    """
    Create one-time bind token for student_id, save pending_tokens.json and links.csv,
    write QR image to qr_links/<student_id>.png if qrcode is installed.
    Returns dict: {"token": token, "link": link, "qr": qr_path or None}
    """
    if not bot_username:
        bot_username = os.getenv("BOT_USERNAME")
    if not bot_username:
        raise RuntimeError("BOT_USERNAME not set in .env and not provided")

    pending = load_pending()

    # reuse existing pending token for this student if requested
    existing = next((t for t, s in pending.items() if s == student_id), None) if reuse else None
    token = existing or gen_token()
    pending[token] = student_id
    save_pending(pending)

    links = load_links()
    link = make_link(bot_username, token)
    links[student_id] = link
    save_links(links)

    qr_path = QR_DIR / f"{student_id}.png"
    ok = _make_qr_image(link, qr_path)
    return {"token": token, "link": link, "qr": str(qr_path) if ok else None}