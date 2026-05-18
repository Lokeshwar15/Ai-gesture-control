import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_ws_accepts_jpeg_and_returns_json():
    ok, buf = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
    assert ok
    frame = buf.tobytes()

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            ws.send_bytes(frame)
            data = ws.receive_json()
            assert "gesture" in data
            assert "confidence" in data
            assert "timestamp" in data
