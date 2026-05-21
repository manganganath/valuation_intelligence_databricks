"""FastAPI application for the Valuation Intelligence Agent."""

import logging
import os
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from agent import call_agent_stream, _extract_memories, _get_auth_headers
from config import PRESET_COMPANIES
from session import add_message, create_session, delete_session, get_history, get_memories, list_sessions, store_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Valuation Intelligence Agent")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main UI."""
    import time
    return templates.TemplateResponse("index.html", {
        "request": request,
        "session_id": "",
        "presets": PRESET_COMPANIES,
        "cache_bust": int(time.time()),
    })


@app.post("/query")
async def query(request: Request):
    """Handle a valuation query — returns SSE stream."""
    body = await request.json()
    user_message = body.get("message", "").strip()
    session_id = body.get("session_id", "")

    if not user_message:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    if not session_id:
        session_id = create_session()

    history = get_history(session_id)
    memories = get_memories()

    def event_generator():
        import json
        full_content = ""
        trace_steps = []
        response_id = ""

        try:
            for event_str in call_agent_stream(user_message, conversation_history=history if history else None, memories=memories):
                yield event_str

                # Parse the event to track state for session storage
                if "event: trace\n" in event_str:
                    data_line = event_str.split("data: ", 1)[1].split("\n")[0]
                    trace_steps.append(json.loads(data_line))
                elif "event: done\n" in event_str:
                    data_line = event_str.split("data: ", 1)[1].split("\n")[0]
                    done_data = json.loads(data_line)
                    full_content = done_data.get("content", "")
                    response_id = done_data.get("response_id", "")
                elif "event: token\n" in event_str:
                    pass  # content accumulated in done event

            # Store in session history after stream completes
            add_message(session_id, "user", user_message)
            if full_content:
                add_message(session_id, "assistant", full_content, trace_steps=trace_steps)

            # Extract and store long-term memories
            if full_content:
                try:
                    headers = _get_auth_headers()
                    insights = _extract_memories(user_message, full_content, headers)
                    for insight in insights:
                        if insight.get("topic") and insight.get("content"):
                            store_memory(insight["topic"], insight["content"])
                except Exception as e:
                    logger.debug(f"Memory extraction skipped: {e}")

            # Send session_id as final event
            yield f"event: session\ndata: {json.dumps({'session_id': session_id})}\n\n"

        except Exception as e:
            logger.error(f"Query error: {traceback.format_exc()}")
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream; charset=utf-8")


@app.get("/health")
async def health():
    from session import _lakebase_available
    return JSONResponse({"lakebase": _lakebase_available})


@app.get("/sessions")
async def sessions_list():
    return JSONResponse(list_sessions())


@app.get("/sessions/{session_id}")
async def session_history(session_id: str):
    history = get_history(session_id)
    return JSONResponse(history)


@app.delete("/sessions/{session_id}")
async def session_delete(session_id: str):
    delete_session(session_id)
    return JSONResponse({"ok": True})
