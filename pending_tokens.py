import os
import json
import csv
import secrets
import string
from pathlib import Path
from dotenv import load_dotenv

from storage_utils import (
    load_json_locked,
    locked_open,
    write_json_locked,
)

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
    data = load_json_locked(PENDING_F, default={})
    if isinstance(data, dict):
        return data
    return {}

def save_pending(d: dict):
    write_json_locked(PENDING_F, d)

def load_links():
    links = {}
    if LINKS_F.exists():
        try:
            with locked_open(LINKS_F, "r", shared=True, newline="") as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    sid = (r.get("student_id") or "").strip()
                    link = (r.get("deep_link") or "").strip()
                    if sid:
                        links[sid] = link
        except Exception:
            pass
    return links

def save_links(links: dict):
    with locked_open(LINKS_F, "w", shared=False, newline="") as f:
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

def create_bind_link_for_student(
    student_id: str, bot_username: str = None, reuse: bool = True, replace_existing: bool = False
):
    """
    Create one-time bind token for student_id, save pending_tokens.json and links.csv,
    write QR image to qr_links/<student_id>.png if qrcode is installed.
    Returns dict: {"token": token, "link": link, "qr": qr_path or None}

    reuse=True will keep an existing pending token for the student.
    replace_existing=True will drop other pending tokens for the student once a new one is created.
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

    if replace_existing:
        to_drop = [t for t, s in pending.items() if s == student_id and t != token]
        for t in to_drop:
            pending.pop(t, None)
    save_pending(pending)

    links = load_links()
    link = make_link(bot_username, token)
    links[student_id] = link
    save_links(links)

    qr_path = QR_DIR / f"{student_id}.png"
    ok = _make_qr_image(link, qr_path)
    return {"token": token, "link": link, "qr": str(qr_path) if ok else None}
