# Step-by-Step Implementation Plan
## AI-Based Gesture Control System

| | |
|---|---|
| **Version** | 1.0 |
| **Date** | May 2026 |
| **Project** | AI Gesture Control System |

---

## Table of Contents

1. [Project Timeline Overview](#1-project-timeline-overview)
2. [Technology Stack](#2-technology-stack)
3. [Phase 1: Foundation (Weeks 1–2)](#3-phase-1-foundation-weeks-12)
4. [Phase 2: Backend Core (Weeks 3–4)](#4-phase-2-backend-core-weeks-34)
5. [Phase 3: Frontend Core (Weeks 5–6)](#5-phase-3-frontend-core-weeks-56)
6. [Phase 4: Integration (Weeks 7–8)](#6-phase-4-integration-weeks-78)
7. [Phase 5: Polish (Weeks 9–10)](#7-phase-5-polish-weeks-910)
8. [Phase 6: Testing (Week 11)](#8-phase-6-testing-week-11)
9. [Phase 7: Deployment (Week 12)](#9-phase-7-deployment-week-12)
10. [Definition of Done](#10-definition-of-done)

---

## 1. Project Timeline Overview

| Phase | Weeks | Focus | Deliverable |
|---|---|---|---|
| Phase 1: Foundation | 1–2 | Environment setup, tooling, project scaffold | Running dev environment |
| Phase 2: Backend Core | 3–4 | FastAPI + MediaPipe gesture engine | Working gesture API |
| Phase 3: Frontend Core | 5–6 | React UI, camera feed, canvas overlay | Live gesture display |
| Phase 4: Integration | 7–8 | WebSocket bridge, action dispatcher | End-to-end gesture control |
| Phase 5: Polish | 9–10 | Settings, mapping UI, error handling | Feature-complete MVP |
| Phase 6: Testing | 11 | Unit, integration, and performance tests | Test reports |
| Phase 7: Deployment | 12 | Dockerise, CI/CD, documentation | Production-ready release |

---

## 2. Technology Stack

### 2.1 Backend Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.11+ | Primary backend language |
| FastAPI | 0.111+ | Async REST API and WebSocket server |
| Uvicorn | 0.29+ | ASGI server to run FastAPI |
| MediaPipe | 0.10+ | Hand, pose, and face landmark detection |
| OpenCV (cv2) | 4.9+ | Frame decoding, image processing, drawing |
| NumPy | 1.26+ | Landmark array manipulation and geometry math |
| PyAutoGUI | 0.9+ | Keyboard and mouse action simulation |
| TensorFlow / Keras | 2.15+ | LSTM model for dynamic gesture classification |
| Redis | 7.x | Session state and gesture event caching (optional) |
| Pydantic | 2.x | Request/response data validation |

### 2.2 Frontend Stack

| Tool | Version | Purpose |
|---|---|---|
| React | 18+ | Component-based frontend framework |
| TypeScript | 5.x | Type-safe JavaScript for the frontend |
| Vite | 5.x | Fast build tool and dev server |
| Tailwind CSS | 3.x | Utility-first styling framework |
| shadcn/ui | Latest | Pre-built accessible UI components |
| Zustand | 4.x | Lightweight global state management |
| React Query | 5.x | API data fetching and caching |
| Lucide React | Latest | Icon library for UI controls |

### 2.3 DevOps & Tooling

| Tool | Purpose |
|---|---|
| Docker + Docker Compose | Containerise backend and frontend services |
| GitHub Actions | CI/CD pipeline for lint, test, and build |
| pytest | Backend unit and integration testing |
| Vitest + React Testing Library | Frontend component and hook testing |
| ESLint + Prettier | Frontend code quality and formatting |
| Ruff + Black | Python code quality and formatting |
| Prometheus + Grafana (optional) | Metrics collection and latency monitoring |

---

## 3. Phase 1: Foundation (Weeks 1–2)

### Step 1.1 — Repository & Project Structure

1. Create GitHub repository with monorepo structure
2. Create `/backend` directory (FastAPI app) and `/frontend` directory (React app)
3. Initialise `.gitignore`, `README.md`, and `CONTRIBUTING.md`
4. Set up pre-commit hooks: `ruff`, `black`, `eslint`, `prettier`

**Folder structure:**

| Path | Contents |
|---|---|
| `backend/app/` | FastAPI application: `main.py`, `routers/`, `services/`, `models/` |
| `backend/app/gesture_engine/` | `mediapipe_processor.py`, `classifier.py`, `action_dispatcher.py` |
| `backend/tests/` | pytest test files |
| `frontend/src/components/` | React components: `CameraFeed`, `GestureOverlay`, `ControlPanel` |
| `frontend/src/hooks/` | `useWebSocket.ts`, `useCamera.ts`, `useGestureStore.ts` |
| `frontend/src/store/` | Zustand stores: `gestureStore.ts`, `settingsStore.ts` |
| `docker/` | `Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml` |

### Step 1.2 — Environment Setup

1. Create Python virtual environment: `python -m venv .venv`
2. Install backend dependencies:
   ```bash
   pip install fastapi uvicorn mediapipe opencv-python pyautogui tensorflow numpy pydantic redis
   ```
3. Initialise React app:
   ```bash
   npm create vite@latest frontend -- --template react-ts
   ```
4. Install frontend dependencies:
   ```bash
   npm install tailwindcss zustand @tanstack/react-query lucide-react
   ```
5. Install shadcn/ui components: `npx shadcn-ui@latest init`
6. Verify MediaPipe installation with a simple hand detection smoke test

---

## 4. Phase 2: Backend Core (Weeks 3–4)

### Step 2.1 — FastAPI Application Scaffold

1. Create `main.py` with FastAPI app, CORS middleware, and lifespan handler
2. Create `/health`, `/gestures`, `/mappings` GET and POST routes
3. Implement Pydantic schemas for `GestureEvent`, `MappingConfig`, `SessionInfo`
4. Write pytest tests for all REST endpoints (happy path + error cases)

### Step 2.2 — MediaPipe Gesture Engine

1. Create `mediapipe_processor.py`: initialise `Hands`, `Pose`, `FaceMesh` solutions
2. Implement `process_frame(frame_bytes) -> LandmarkResult` function
3. Decode JPEG bytes to numpy array using `cv2.imdecode`
4. Pass array through `mp.solutions.hands.Hands.process()`
5. Extract landmark x/y/z coordinates into structured dict
6. Implement `get_finger_states(landmarks)` helper using angle threshold logic

### Step 2.3 — Gesture Classifier

1. Implement rule-based static gesture classifier using finger state combinations
2. Define gesture rules: `thumbs_up`, `open_palm`, `closed_fist`, `peace`, `pinch` etc.
3. Implement 30-frame sliding window buffer for dynamic gesture input
4. Train lightweight LSTM model on a collected swipe/wave dataset (or use pretrained)
5. Export LSTM model as `.keras` file and load in `classifier.py` on startup
6. Implement `classify(landmarks, history) -> GestureResult` with confidence score

### Step 2.4 — WebSocket Server

1. Add `/ws` WebSocket endpoint to FastAPI app
2. Implement frame receive loop: accept binary message, decode, process, respond
3. Serialise `GestureEvent` to JSON and send back over WebSocket
4. Add connection manager to track active sessions
5. Implement 30-second inactivity timeout and graceful close

### Step 2.5 — Action Dispatcher

1. Create `action_dispatcher.py` with `DEFAULT_MAPPINGS` dict
2. Implement `dispatch(gesture_label)` function using `pyautogui`
3. Add configurable debounce: same gesture must not fire more than once per 500ms
4. Load user mappings from `mappings.json` config file on startup
5. Expose `update_mapping(gesture, action)` method for runtime remapping

---

## 5. Phase 3: Frontend Core (Weeks 5–6)

### Step 3.1 — Camera Feed Component

1. Create `useCamera.ts` hook: call `navigator.mediaDevices.getUserMedia`
2. Bind `MediaStream` to `<video>` element ref with `autoPlay` and `muted`
3. Mirror video horizontally with CSS: `transform: scaleX(-1)`
4. Implement start/stop/toggle controls with state management
5. Handle permission denied and device not found error states

### Step 3.2 — Gesture Overlay Component

1. Create `<canvas>` element absolutely positioned over the video element
2. Implement `drawLandmarks(ctx, landmarks)` function
3. Draw 21 hand landmark points as filled circles (colour-coded by finger)
4. Draw finger connections using `HAND_CONNECTIONS` constant from MediaPipe spec
5. Implement `drawGestureBadge(ctx, gesture, confidence)` for pill label overlay
6. Run draw loop via `requestAnimationFrame`, clearing canvas each frame

### Step 3.3 — WebSocket Hook

1. Create `useWebSocket.ts` hook: connect to `ws://localhost:8000/ws` on mount
2. Implement `sendFrame(videoElement)`: capture frame to canvas, export as Blob, send
3. Parse incoming JSON messages and update Zustand gesture store
4. Implement auto-reconnect: `useEffect` re-runs on close with exponential backoff
5. Expose `connectionStatus: 'connected' | 'reconnecting' | 'disconnected'`

### Step 3.4 — Control Panel Component

1. Build `GestureRow` component: icon + gesture name + action dropdown + enable toggle
2. Fetch `/mappings` on mount with React Query
3. Optimistically update UI on mapping change, POST `/mappings` to backend
4. Build `GestureHistory` list: last 10 events with gesture name and timestamp
5. Highlight currently active gesture row with animation

---

## 6. Phase 4: Integration (Weeks 7–8)

### Step 4.1 — End-to-End Wiring

1. Start backend: `uvicorn app.main:app --reload --port 8000`
2. Start frontend: `npm run dev` (Vite serves on port 5173)
3. Verify WebSocket connection opens on frontend load
4. Verify frames flow: camera → canvas → WebSocket → backend → MediaPipe → response
5. Verify gesture label and confidence appear in overlay and control panel

### Step 4.2 — Action Dispatch Verification

1. Test each default gesture mapping triggers the correct `pyautogui` action
2. Verify debounce prevents repeated triggers on held gestures
3. Test mapping remapping: change action in UI, verify new action fires
4. Test `UNKNOWN` gesture returns no action dispatch

### Step 4.3 — Latency Measurement

1. Add timing logs: `frame_received_at`, `landmarks_computed_at`, `response_sent_at`
2. Measure round-trip from frame capture to canvas overlay update
3. Optimise if latency > 200ms: reduce frame resolution, skip every other frame

---

## 7. Phase 5: Polish (Weeks 9–10)

### Step 5.1 — Settings Panel

1. Build Settings modal with resolution, mode, confidence, and hold-duration controls
2. Persist settings to `localStorage` on change
3. Apply settings to `useCamera` and `useWebSocket` hooks reactively

### Step 5.2 — Error Handling & UX

1. Add toast notifications for connection status changes
2. Add onboarding overlay for first-time users: camera setup guide
3. Add visual pulse animation on gesture trigger
4. Add confidence meter progress bar below overlay badge
5. Implement dark/light theme toggle with Tailwind `dark:` classes

---

## 8. Phase 6: Testing (Week 11)

| Test Type | Tool | Coverage Target | Focus Areas |
|---|---|---|---|
| Backend unit tests | pytest | >= 80% | Gesture classifier, dispatcher, frame processing |
| Backend API tests | pytest + httpx | >= 90% | All REST endpoints, error responses |
| WebSocket tests | pytest + websockets | >= 75% | Frame handling, timeout, multi-client |
| Frontend unit tests | Vitest | >= 70% | Hooks, store actions, utility functions |
| Frontend component tests | React Testing Library | >= 70% | CameraFeed, ControlPanel, GestureOverlay |
| Integration tests | Playwright | Key flows | Full gesture → action end-to-end |
| Performance tests | Custom benchmark | Latency < 200ms | Frame rate, WebSocket throughput |

---

## 9. Phase 7: Deployment (Week 12)

### Step 7.1 — Dockerisation

1. Create `Dockerfile.backend`: `python:3.11-slim` base, copy app, pip install, `CMD uvicorn`
2. Create `Dockerfile.frontend`: `node:18-alpine` build stage, `nginx` serve stage
3. Create `docker-compose.yml`: backend service (port 8000), frontend service (port 80), optional redis service
4. Test full stack with `docker-compose up`

### Step 7.2 — CI/CD Pipeline

1. Create `.github/workflows/ci.yml`
2. Jobs: `lint-backend` (ruff + black), `lint-frontend` (eslint + prettier), `test-backend` (pytest), `test-frontend` (vitest), `build-docker`
3. Add branch protection: `main` requires all CI jobs to pass

### Step 7.3 — Documentation

1. Write `README.md`: project overview, quick start, environment variables, gesture list
2. Write API reference in `docs/api.md` using FastAPI auto-generated OpenAPI spec
3. Write `CONTRIBUTING.md`: branching strategy, PR template, code style guide
4. Record demo video showing gesture control in action

---

## 10. Definition of Done

| Milestone | Done When |
|---|---|
| Phase 1 complete | Dev environment runs; project structure committed; CI lint passes |
| Phase 2 complete | Backend detects gestures from webcam; all REST tests pass |
| Phase 3 complete | Frontend shows live camera feed with landmark overlay and gesture badge |
| Phase 4 complete | Gestures trigger mapped actions end-to-end; latency < 200ms |
| Phase 5 complete | Settings panel works; error states handled; onboarding flow complete |
| Phase 6 complete | All test suites pass; coverage targets met; perf benchmark passes |
| Phase 7 complete | Docker Compose runs full stack; CI/CD green; README complete |
