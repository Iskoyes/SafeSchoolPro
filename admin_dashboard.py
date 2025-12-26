import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, flash, redirect, render_template, request, url_for

from parents_utils import load_parents, remove_chat_binding
from pending_tokens import (
    create_bind_link_for_student,
    load_links,
    load_pending,
    save_links,
    save_pending,
)

# Load environment from current dir, fallback to safeschool-recognition/.env
load_dotenv()
if not os.getenv("BOT_USERNAME") or not os.getenv("TELEGRAM_BOT_TOKEN"):
    alt = Path(__file__).parent / "safeschool-recognition" / ".env"
    if alt.exists():
        load_dotenv(str(alt))

BOT_USERNAME = (os.getenv("BOT_USERNAME") or "").strip().lstrip("@")
TELEGRAM_BOT_TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TOKEN") or "").strip()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "safeschool-admin")


def build_bot_status():
    errors = []
    if not BOT_USERNAME:
        errors.append("BOT_USERNAME is missing in the environment.")
    if not TELEGRAM_BOT_TOKEN:
        errors.append("TELEGRAM_BOT_TOKEN (or TOKEN) is missing in the environment.")
    deep_link_format = f"https://t.me/{BOT_USERNAME or '<bot_username>'}?start=<token>"
    return {
        "username": BOT_USERNAME,
        "token_set": bool(TELEGRAM_BOT_TOKEN),
        "errors": errors,
        "deep_link_format": deep_link_format,
    }


def _collect_student_rows(parents, pending, links):
    students = set(parents.keys()) | set(links.keys()) | set(pending.values())
    rows = []
    for sid in sorted(students):
        rows.append(
            {
                "student_id": sid,
                "chats": parents.get(sid, []),
                "tokens": [t for t, s in pending.items() if s == sid],
                "link": links.get(sid, ""),
            }
        )
    return rows


@app.route("/")
def index():
    parents = load_parents()
    pending = load_pending()
    links = load_links()
    token_rows = sorted(pending.items(), key=lambda kv: kv[0])
    student_rows = _collect_student_rows(parents, pending, links)
    bot_status = build_bot_status()
    return render_template(
        "admin_dashboard.html",
        bot_status=bot_status,
        token_rows=token_rows,
        student_rows=student_rows,
    )


@app.post("/revoke_token")
def revoke_token():
    token = (request.form.get("token") or "").strip()
    pending = load_pending()
    if token in pending:
        pending.pop(token, None)
        save_pending(pending)
        flash(f"Token {token} revoked.")
    else:
        flash("Token not found.", "error")
    return redirect(url_for("index"))


@app.post("/unbind_chat")
def unbind_chat():
    student_id = (request.form.get("student_id") or "").strip()
    chat_id_raw = (request.form.get("chat_id") or "").strip()
    try:
        chat_id = int(chat_id_raw)
    except ValueError:
        flash("Invalid chat id.", "error")
        return redirect(url_for("index"))

    ok = remove_chat_binding(student_id, chat_id)
    if ok:
        flash(f"Chat {chat_id} unbound from {student_id}.")
    else:
        flash("Binding not found.", "error")
    return redirect(url_for("index"))


@app.post("/regenerate_link")
def regenerate_link():
    student_id = (request.form.get("student_id") or "").strip()
    if not student_id:
        flash("Student ID is required to regenerate a link.", "error")
        return redirect(url_for("index"))

    try:
        res = create_bind_link_for_student(
            student_id, bot_username=BOT_USERNAME or None, reuse=False, replace_existing=True
        )
    except Exception as e:
        flash(f"Failed to regenerate link: {e}", "error")
        return redirect(url_for("index"))

    # Ensure links.csv reflects the newly generated link (create_bind_link already writes it).
    links = load_links()
    links[student_id] = res["link"]
    save_links(links)

    flash(f"New link created for {student_id}: {res['link']}")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
