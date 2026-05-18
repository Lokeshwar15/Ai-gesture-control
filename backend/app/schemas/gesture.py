from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float


class HandLandmarks(BaseModel):
    """21 MediaPipe hand landmarks in image-normalized coordinates."""

    points: List[LandmarkPoint] = Field(..., min_length=21, max_length=21)
    handedness: str = "Unknown"  # "Left" or "Right"


class GestureResult(BaseModel):
    gesture: str
    confidence: float = Field(ge=0.0, le=1.0)
    landmarks: Optional[List[Dict[str, Any]]] = None
    action: Optional[str] = None


class GestureEvent(BaseModel):
    gesture: str
    confidence: float
    landmarks: Optional[List[Dict[str, Any]]] = None
    action: Optional[str] = None
    timestamp: float
