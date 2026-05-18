"""FastAPI entry: REST + WebSocket gesture pipeline."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.gesture_engine.action_dispatcher import (
    ACTION_DESCRIPTIONS,
    DEFAULT_MAPPINGS,
    ActionDispatcher,
)
from app.gesture_engine.classifier import GestureClassifierSession
from app.gesture_engine.dynamic_classifier import DynamicClassifier
from app.gesture_engine.mediapipe_processor import (
    close_mediapipe,
    landmarks_to_serializable,
    process_jpeg_bytes,
)
from app.schemas.gesture import GestureEvent

GESTURE_DESCRIPTIONS: dict[str, str] = {
    "thumbs_up": "Thumb up, others curled",
    "thumbs_down": "Thumb down, others curled",
    "open_palm": "All fingers extended",
    "closed_fist": "All fingers curled",
    "peace_sign": "Index and middle extended",
    "pinch": "Thumb and index pinch",
    "swipe_left": "Palm swipe left",
    "swipe_right": "Palm swipe right",
    "swipe_up": "Palm swipe up",
    "wave": "Wrist wave oscillation",
    "UNKNOWN": "No confident gesture",
}


class MappingUpdate(BaseModel):
    gesture: str
    action: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.dynamic = DynamicClassifier()
    app.state.history: list[dict[str, Any]] = []
    app.state.sessions: dict[str, dict[str, Any]] = {}
    yield
    close_mediapipe()


app = FastAPI(title="Gesture Control API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _record_history(payload: dict[str, Any]) -> None:
    hist: list = app.state.history
    hist.append(payload)
    if len(hist) > 1000:
        del hist[: len(hist) - 1000]


@app.get("/health")
def health() -> dict[str, Any]:
    dyn: DynamicClassifier = app.state.dynamic
    return {
        "status": "ok",
        "lstm_loaded": dyn.model_loaded,
        "timestamp": time.time(),
    }


@app.get("/gestures")
def list_gestures() -> dict[str, Any]:
    gestures = []
    for name in sorted(set(DEFAULT_MAPPINGS.keys()) | set(GESTURE_DESCRIPTIONS.keys())):
        if name == "UNKNOWN":
            continue
        gestures.append(
            {
                "id": name,
                "description": GESTURE_DESCRIPTIONS.get(name, name),
                "default_action": DEFAULT_MAPPINGS.get(name),
                "action_description": ACTION_DESCRIPTIONS.get(
                    DEFAULT_MAPPINGS.get(name, ""), ""
                ),
            }
        )
    return {"gestures": gestures}


@app.get("/mappings")
def get_mappings() -> dict[str, str]:
    d = ActionDispatcher(dry_run=True)
    return d.get_mappings()


@app.post("/mappings")
def post_mappings(body: MappingUpdate) -> dict[str, str]:
    d = ActionDispatcher(dry_run=True)
    d.update_mapping(body.gesture, body.action)
    return d.get_mappings()


@app.post("/session/start")
def session_start() -> dict[str, str]:
    sid = str(uuid.uuid4())
    app.state.sessions[sid] = {"started_at": time.time()}
    return {"session_id": sid}


@app.post("/session/stop")
def session_stop() -> dict[str, str]:
    app.state.sessions.clear()
    return {"status": "stopped"}


@app.get("/history")
def get_history(limit: int = 50) -> dict[str, Any]:
    items = app.state.history[-limit:]
    return {"items": items, "total": len(app.state.history)}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    dyn: DynamicClassifier = app.state.dynamic
    session = GestureClassifierSession(dynamic=dyn)
    dispatcher = ActionDispatcher(dry_run=False)

    try:
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.close(code=1000)
                break
            hands, wh = process_jpeg_bytes(msg)
            ser = landmarks_to_serializable(hands)
            result = session.classify(hands, wh, ser)
            action = dispatcher.dispatch(result.gesture, result.confidence)
            event = GestureEvent(
                gesture=result.gesture,
                confidence=result.confidence,
                landmarks=result.landmarks,
                action=action,
                timestamp=time.time(),
            )
            payload = event.model_dump()
            await websocket.send_json(payload)
            if result.gesture != "UNKNOWN":
                _record_history(payload)
    except WebSocketDisconnect:
        return
    except Exception as e:  # noqa: BLE001
        await websocket.close(code=1011, reason=str(e)[:120])
