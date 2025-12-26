import asyncio
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from recognition.settings import RecognitionSettings, SettingsStore
from recognition.stream import RecognitionStream

BASE_DIR = Path(__file__).resolve().parent.parent
settings_store = SettingsStore(BASE_DIR / "settings.json")
stream = RecognitionStream(settings_store)
stream.start()

app = FastAPI(title="SafeSchool Recognition UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _render_page(initial_settings: RecognitionSettings) -> str:
    return f"""
    <!doctype html>
    <html lang="ru">
    <head>
        <meta charset="utf-8" />
        <title>SafeSchool Recognition</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 16px; background: #f4f6f8; }}
            h1 {{ margin-top: 0; }}
            .grid {{ display: grid; grid-template-columns: 2fr 1fr; gap: 16px; }}
            .card {{ background: white; border-radius: 8px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            label {{ display: block; margin: 6px 0 2px; font-weight: 600; }}
            input {{ width: 100%; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; }}
            button {{ margin-right: 8px; padding: 8px 12px; border: none; border-radius: 4px; cursor: pointer; }}
            button.primary {{ background: #2563eb; color: white; }}
            button.secondary {{ background: #f97316; color: white; }}
            .muted {{ color: #6b7280; font-size: 13px; }}
            ul {{ list-style: none; padding-left: 0; }}
            li {{ padding: 6px 0; border-bottom: 1px solid #eee; }}
            .pill {{ display: inline-block; padding: 2px 6px; border-radius: 10px; font-size: 12px; margin-left: 6px; }}
            .pill.success {{ background: #dcfce7; color: #15803d; }}
            .pill.error {{ background: #fee2e2; color: #b91c1c; }}
            .status {{ margin-top: 6px; font-size: 14px; }}
        </style>
    </head>
    <body>
        <h1>SafeSchool Recognition</h1>
        <div class="status">Статус: <span id="status-text">загружается...</span></div>
        <div class="grid">
            <div class="card">
                <h2>Поток</h2>
                <img id="stream" src="/video" style="width:100%; border-radius: 6px; background: #111;" />
            </div>
            <div class="card">
                <h2>Управление</h2>
                <div>
                    <button class="primary" onclick="startStream()">Запустить</button>
                    <button class="secondary" onclick="stopStream()">Остановить</button>
                </div>
                <form id="settings-form" style="margin-top: 12px;">
                    <label>Источник камеры</label>
                    <input type="text" name="camera_source" value="{initial_settings.camera_source}">
                    <span class="muted">Оставьте пустым для локальной камеры 0</span>
                    <label>PROCESS_EVERY_N</label>
                    <input type="number" min="1" step="1" name="process_every_n" value="{initial_settings.process_every_n}">
                    <label>SIM_THRESHOLD</label>
                    <input type="number" step="0.01" name="sim_threshold" value="{initial_settings.sim_threshold}">
                    <button type="submit" class="primary" style="margin-top: 10px;">Сохранить параметры</button>
                </form>
                <div id="errors" class="muted" style="margin-top:8px;"></div>
            </div>
        </div>
        <div class="grid" style="margin-top: 16px;">
            <div class="card">
                <h2>Последние срабатывания</h2>
                <ul id="events"></ul>
            </div>
            <div class="card">
                <h2>Статус Telegram</h2>
                <ul id="telegram"></ul>
            </div>
        </div>
        <script>
            const eventsList = document.getElementById('events');
            const telegramList = document.getElementById('telegram');
            const statusText = document.getElementById('status-text');
            const errorsBlock = document.getElementById('errors');

            function renderEvent(event) {{
                const li = document.createElement('li');
                li.textContent = `${{event.timestamp}} · ${{event.name}} (sim: ${{event.similarity.toFixed(2)}})`;
                eventsList.prepend(li);
                while (eventsList.children.length > 20) {{
                    eventsList.removeChild(eventsList.lastChild);
                }}
            }}

            function renderTelegram(status) {{
                const li = document.createElement('li');
                const pill = document.createElement('span');
                pill.className = 'pill ' + (status.success ? 'success' : 'error');
                pill.textContent = status.success ? 'ok' : 'error';
                li.textContent = `${{status.timestamp}} · ${{status.message}} `;
                li.appendChild(pill);
                telegramList.prepend(li);
                while (telegramList.children.length > 20) {{
                    telegramList.removeChild(telegramList.lastChild);
                }}
            }}

            async function refreshStatus() {{
                const resp = await fetch('/api/status');
                const data = await resp.json();
                statusText.textContent = data.running ? 'работает' : 'остановлен';
                errorsBlock.textContent = data.last_error || '';
                eventsList.innerHTML = '';
                telegramList.innerHTML = '';
                data.recent_events.forEach(renderEvent);
                data.telegram.forEach(renderTelegram);
            }}

            async function startStream() {{
                await fetch('/api/start', {{ method: 'POST' }});
                refreshStatus();
            }}
            async function stopStream() {{
                await fetch('/api/stop', {{ method: 'POST' }});
                refreshStatus();
            }}

            document.getElementById('settings-form').addEventListener('submit', async (e) => {{
                e.preventDefault();
                const form = e.target;
                const payload = {{
                    camera_source: form.camera_source.value,
                    process_every_n: parseInt(form.process_every_n.value || '1', 10),
                    sim_threshold: parseFloat(form.sim_threshold.value || '0.38')
                }};
                await fetch('/api/settings', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(payload),
                }});
                refreshStatus();
            }});

            function connectWS() {{
                const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
                const ws = new WebSocket(`${{protocol}}://${{location.host}}/ws`);
                ws.onmessage = (event) => {{
                    const payload = JSON.parse(event.data);
                    if (payload.source === 'telegram') {{
                        renderTelegram(payload.data);
                        return;
                    }}
                    if (payload.type === 'event') {{
                        renderEvent(payload.data);
                    }} else if (payload.type === 'status') {{
                        const status = payload.data || {{}};
                        if (typeof status.running !== 'undefined') {{
                            statusText.textContent = status.running ? 'работает' : 'остановлен';
                        }}
                        if (status.message) {{
                            errorsBlock.textContent = status.message;
                        }}
                        if (typeof status.success !== 'undefined') {{
                            renderTelegram(status);
                        }}
                    }}
                }};
                ws.onclose = () => setTimeout(connectWS, 1500);
            }}

            refreshStatus();
            connectWS();
        </script>
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(_render_page(stream.settings))


@app.get("/api/status")
async def api_status():
    return JSONResponse(stream.status())


@app.post("/api/start")
async def api_start():
    stream.start()
    return JSONResponse({"running": True})


@app.post("/api/stop")
async def api_stop():
    stream.stop()
    return JSONResponse({"running": False})


@app.post("/api/settings")
async def api_settings(payload: Dict[str, Any]):
    updates = {}
    if "camera_source" in payload:
        updates["camera_source"] = payload["camera_source"]
    if "process_every_n" in payload:
        try:
            updates["process_every_n"] = int(payload["process_every_n"])
        except Exception:
            pass
    if "sim_threshold" in payload:
        try:
            updates["sim_threshold"] = float(payload["sim_threshold"])
        except Exception:
            pass
    if updates:
        stream.update_settings(**updates)
        if stream.status().get("running"):
            stream.stop()
            stream.start()
    return JSONResponse({"updated": updates})


@app.get("/video")
async def video_stream():
    if not stream.status().get("running"):
        stream.start()

    async def frame_iterator():
        while True:
            chunk = await asyncio.to_thread(stream.next_frame, 1.0)
            if chunk is None:
                if not stream.status().get("running"):
                    await asyncio.sleep(0.5)
                else:
                    await asyncio.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"

    return StreamingResponse(frame_iterator(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "status", "data": stream.status(), "source": "runtime"})
    try:
        while True:
            msg = await asyncio.to_thread(stream.messages, 1.0)
            if msg:
                await ws.send_json(msg)
            else:
                await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        return
