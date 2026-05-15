# Functional Requirements Document
## AI-Based Gesture Control System

| | |
|---|---|
| **Version** | 1.0 |
| **Date** | May 2026 |
| **Project** | AI Gesture Control System |

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [System Architecture](#2-system-architecture)
3. [Frontend Functional Requirements](#3-frontend-functional-requirements)
4. [Backend Functional Requirements](#4-backend-functional-requirements)
5. [Gesture Specification](#5-gesture-specification)
6. [Error Handling](#6-error-handling)
7. [UI Screen Specifications](#7-ui-screen-specifications)
8. [Performance Requirements](#8-performance-requirements)

---

## 1. System Overview

This Functional Requirements Document (FRD) specifies the detailed functional behaviour of the AI Gesture Control System. It covers the frontend React application, the backend FastAPI service, the WebSocket event bus, and the MediaPipe/OpenCV gesture processing pipeline. This document is intended for the engineering team and serves as the technical contract for implementation.

---

## 2. System Architecture

### 2.1 High-Level Components

| Component | Technology | Responsibility |
|---|---|---|
| Frontend UI | React 18 + TypeScript | Camera feed display, gesture overlay, action panel |
| WebSocket Client | Browser WebSocket API | Real-time bidirectional event stream with backend |
| Backend API | FastAPI (Python 3.11) | REST endpoints, gesture session management |
| WebSocket Server | FastAPI WebSockets | Push gesture events to connected frontend clients |
| Gesture Engine | MediaPipe + OpenCV | Landmark detection and gesture classification |
| Action Dispatcher | Python event bus | Map gesture labels to application actions |
| State Store | Redis (optional) | Session state and gesture history cache |

### 2.2 Data Flow

1. Browser accesses webcam via MediaDevices API
2. Video frames captured at 30fps and displayed in canvas element
3. Frame bytes streamed over WebSocket to FastAPI backend
4. Backend passes frame to MediaPipe for landmark extraction
5. Landmark coordinates fed to gesture classifier (rule-based or ML model)
6. Gesture label and confidence score returned over WebSocket to frontend
7. Frontend renders overlay on canvas and triggers mapped action
8. Action dispatcher fires keyboard/mouse simulation or custom callback

---

## 3. Frontend Functional Requirements

### 3.1 Camera Feed Module

| ID | Requirement |
|---|---|
| FE-01 | Application shall request camera permission on load using getUserMedia API |
| FE-02 | Live camera feed shall be rendered in a `<video>` element mirrored horizontally |
| FE-03 | A `<canvas>` element overlaid on the video shall render gesture landmarks in real time |
| FE-04 | Frame capture shall operate at 30fps using requestAnimationFrame loop |
| FE-05 | User shall be able to pause/resume the camera feed via a toggle button |
| FE-06 | Camera resolution shall default to 1280x720; user can select 640x480 for performance |

### 3.2 Gesture Overlay Module

| ID | Requirement |
|---|---|
| FE-07 | Hand landmarks (21 points per hand) shall be drawn as connected dots on the canvas |
| FE-08 | Each finger connection shall be drawn with a distinct colour per finger group |
| FE-09 | Detected gesture label shall appear as a pill badge above the hand bounding box |
| FE-10 | Confidence score (0–100%) shall be displayed alongside the gesture label |
| FE-11 | Overlay shall update every frame without canvas flicker or ghosting |
| FE-12 | Body pose skeleton (33 points) shall optionally render when pose mode is enabled |

### 3.3 Gesture Control Panel

| ID | Requirement |
|---|---|
| FE-13 | A side panel shall list all active gesture-to-action mappings |
| FE-14 | Each mapping entry shall show the gesture icon, gesture name, and mapped action |
| FE-15 | User shall be able to reassign a gesture to a different action via a dropdown |
| FE-16 | Active gesture shall be highlighted in the control panel in real time |
| FE-17 | A history log shall display the last 10 detected gestures with timestamps |
| FE-18 | User shall be able to enable/disable individual gesture mappings with a toggle |

### 3.4 WebSocket Client

| ID | Requirement |
|---|---|
| FE-19 | Frontend shall open a WebSocket connection to `ws://localhost:8000/ws` on load |
| FE-20 | Each captured frame shall be sent as a binary message (JPEG blob) over WebSocket |
| FE-21 | Frontend shall parse JSON responses: `{ gesture, confidence, landmarks, action }` |
| FE-22 | Connection status indicator shall show Connected / Reconnecting / Disconnected |
| FE-23 | Frontend shall auto-reconnect with exponential backoff on connection loss |

---

## 4. Backend Functional Requirements

### 4.1 REST API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — returns system status and model load state |
| GET | `/gestures` | Returns list of all supported gestures with descriptions |
| GET | `/mappings` | Returns current gesture-to-action mapping configuration |
| POST | `/mappings` | Update gesture-to-action mapping (body: `{ gesture, action }`) |
| GET | `/history` | Returns paginated gesture event history |
| POST | `/session/start` | Initialise a new gesture session and return session ID |
| POST | `/session/stop` | Terminate active session and flush history |

### 4.2 WebSocket Server

| ID | Requirement |
|---|---|
| BE-01 | Server shall accept WebSocket connections at `/ws` endpoint |
| BE-02 | Server shall decode incoming JPEG binary frames to numpy arrays via OpenCV |
| BE-03 | Server shall pass each frame through the MediaPipe Hands and Pose pipelines |
| BE-04 | Server shall extract landmark coordinates and compute gesture classification |
| BE-05 | Server shall respond with JSON: `{ gesture, confidence, landmarks[], action, timestamp }` |
| BE-06 | Server shall support up to 50 concurrent WebSocket connections |
| BE-07 | Server shall close stale connections after 30 seconds of inactivity |

### 4.3 Gesture Engine

| ID | Requirement |
|---|---|
| GE-01 | Engine shall use MediaPipe Hands model (`max_num_hands=2`) for hand detection |
| GE-02 | Engine shall use MediaPipe Pose model for full-body landmark detection |
| GE-03 | Engine shall compute finger state (extended/curled) from landmark angles |
| GE-04 | Static gestures shall be classified by a rule-based classifier on landmark geometry |
| GE-05 | Dynamic gestures shall be classified by an LSTM model on a 30-frame sliding window |
| GE-06 | Engine shall return `UNKNOWN` when confidence is below configurable threshold (default 0.75) |
| GE-07 | Engine shall support hot-swappable gesture classifier modules |

### 4.4 Action Dispatcher — Default Mappings

| Gesture | Default Action | Method |
|---|---|---|
| Thumbs Up | Volume Up (+5%) | `pyautogui.press('volumeup')` |
| Thumbs Down | Volume Down (-5%) | `pyautogui.press('volumedown')` |
| Open Palm | Pause / Play Media | `pyautogui.press('playpause')` |
| Closed Fist | Mute Toggle | `pyautogui.hotkey('ctrl', 'm')` |
| Swipe Left | Previous Slide / Track | `pyautogui.press('left')` |
| Swipe Right | Next Slide / Track | `pyautogui.press('right')` |
| Pinch | Zoom In | `pyautogui.hotkey('ctrl', '+')` |
| Spread | Zoom Out | `pyautogui.hotkey('ctrl', '-')` |
| Index Point Up | Scroll Up | `pyautogui.scroll(3)` |
| Peace Sign | Screenshot | `pyautogui.hotkey('ctrl', 'shift', 's')` |

---

## 5. Gesture Specification

| Gesture Name | Type | Detection Method | Trigger Condition |
|---|---|---|---|
| Thumbs Up | Static | Thumb extended, all others curled | Hold for 300ms |
| Thumbs Down | Static | Thumb pointing down, fingers curled | Hold for 300ms |
| Open Palm | Static | All 5 fingers extended | Hold for 200ms |
| Closed Fist | Static | All 5 fingers curled | Hold for 300ms |
| Peace Sign | Static | Index + middle extended, others curled | Hold for 200ms |
| Pinch | Static | Thumb + index tip distance < 30px | Hold for 200ms |
| Swipe Left | Dynamic | Palm velocity > 800px/s leftward | Single motion |
| Swipe Right | Dynamic | Palm velocity > 800px/s rightward | Single motion |
| Swipe Up | Dynamic | Palm velocity > 800px/s upward | Single motion |
| Wave | Dynamic | Wrist oscillation > 3 cycles/sec | 2+ oscillations |

---

## 6. Error Handling

| Error Condition | Frontend Behaviour | Backend Behaviour |
|---|---|---|
| Camera not found | Show error banner with setup guide | N/A |
| Camera permission denied | Show permission prompt with instructions | N/A |
| WebSocket connection failed | Show Disconnected badge, retry with backoff | Log connection error |
| No hand detected in frame | Show "No gesture detected" in overlay | Return null gesture response |
| Low confidence gesture | Show gesture label with grey badge (uncertain) | Return UNKNOWN label |
| Backend crash / restart | Auto-reconnect after 2s, 4s, 8s | Restart uvicorn process |

---

## 7. UI Screen Specifications

### 7.1 Main Dashboard Layout

- **Left panel (70% width):** Live camera feed with canvas overlay
- **Right panel (30% width):** Gesture control panel with mappings and history
- **Top bar:** App title, connection status, settings icon, camera toggle
- **Bottom bar:** Detected gesture name, confidence meter, active action label

### 7.2 Settings Modal

- Camera resolution selector (640×480, 1280×720, 1920×1080)
- Detection mode selector (Hands Only, Pose Only, Holistic)
- Confidence threshold slider (0.50 – 0.95)
- Gesture hold duration slider (100ms – 1000ms)
- Dark/Light theme toggle

---

## 8. Performance Requirements

| Metric | Target | Notes |
|---|---|---|
| Frame processing latency | < 150ms | Measured backend-side on CPU |
| WebSocket round-trip latency | < 200ms | Frame send to action trigger |
| Frontend render FPS | >= 25 FPS | Canvas overlay update rate |
| Gesture accuracy (static) | >= 95% | On curated validation set |
| Gesture accuracy (dynamic) | >= 88% | LSTM model on validation set |
| Max concurrent sessions | 50 | FastAPI + Redis session store |
