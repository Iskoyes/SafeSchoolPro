import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from telegram.ext import Updater, CommandHandler

# load .env from cwd, fallback to safeschool-recognition/.env
load_dotenv()
if not os.getenv("TELEGRAM_BOT_TOKEN"):
    alt = Path(__file__).parent / "safeschool-recognition" / ".env"
    if alt.exists():
        load_dotenv(str(alt))

TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TOKEN") or "").strip().strip('"').strip("'")
BOT_USERNAME = (os.getenv("BOT_USERNAME") or "").strip().strip('"').strip("'")

PENDING_F = Path("pending_tokens.json")
PARENTS_F = Path("parents.json")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error("Failed to read %s: %s", path, e)
            return {}
    return {}

def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def start(update, context):
    """Handle /start <token> deep link binding."""
    args = context.args
    if not args:
        update.message.reply_text(
            "Привет! Чтобы привязать себя к ученику, откройте ссылку из школы или отправьте:\n"
            "/start bind-<токен>"
        )
        return

    token = args[0].strip().strip('"').strip("'")
    pending = load_json(PENDING_F)
    if token not in pending:
        update.message.reply_text("Неверный или просроченный токен.")
        logger.info("Invalid token attempt: %s from chat %s", token, update.effective_chat.id)
        return

    student_id = pending[token]
    parents = load_json(PARENTS_F)
    chat_id = update.effective_chat.id

    parents.setdefault(student_id, [])
    if chat_id not in parents[student_id]:
        parents[student_id].append(chat_id)
        save_json(PARENTS_F, parents)
        update.message.reply_text(f"Готово. Вы привязаны к ученику: {student_id}")
        logger.info("Bound chat %s -> %s", chat_id, student_id)
    else:
        update.message.reply_text(f"Вы уже привязаны к ученику: {student_id}")

    # make token one-time
    try:
        del pending[token]
        save_json(PENDING_F, pending)
    except Exception as e:
        logger.error("Failed to remove token %s: %s", token, e)

def info(update, context):
    """Show students bound to this chat."""
    parents = load_json(PARENTS_F)
    chat_id = update.effective_chat.id
    bound = [sid for sid, chats in parents.items() if chat_id in chats]
    if not bound:
        update.message.reply_text("Вы не привязаны ни к одному ученику.")
    else:
        update.message.reply_text("Вы привязаны к:\n" + "\n".join(bound))

def unbind(update, context):
    """Unbind this chat from a student: /unbind <student_id>"""
    args = context.args
    if not args:
        update.message.reply_text("Использование: /unbind <student_id>")
        return
    sid = args[0].strip()
    parents = load_json(PARENTS_F)
    chat_id = update.effective_chat.id
    if sid in parents and chat_id in parents[sid]:
        parents[sid] = [c for c in parents[sid] if c != chat_id]
        if not parents[sid]:
            del parents[sid]
        save_json(PARENTS_F, parents)
        update.message.reply_text(f"Вы отвязаны от {sid}")
    else:
        update.message.reply_text("Вы не привязаны к указанному ученику.")

if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN не задан в .env")
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("info", info))
    dp.add_handler(CommandHandler("unbind", unbind))

    logger.info("Bot starting...")
    updater.start_polling()
    updater.idle()