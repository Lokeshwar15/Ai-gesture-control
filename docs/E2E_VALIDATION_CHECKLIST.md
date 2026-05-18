# End-to-end validation checklist (local co-located)

1. From `backend/`: `python -m uvicorn app.main:app --reload --port 8000`
2. From `frontend/`: `npm install && npm run dev`
3. Open `http://localhost:5173`, allow camera and confirm WebSocket shows **connected**
4. Verify live landmarks overlay tracks your hand
5. Perform each static gesture (thumbs up/down, open palm, fist, peace, pinch) and confirm the on-screen label matches
6. With a trained `ml/models/gesture_lstm.keras`, perform swipe/wave motions and confirm dynamic labels
7. Confirm OS actions fire once per gesture (debounce ~500ms)
8. Edge cases: cover camera (UNKNOWN), deny permission (error banner), stop/start backend (reconnecting)
9. Run `python ml/scripts/evaluate_model.py --no-gate` and `python ml/scripts/benchmark_latency.py` from `backend/`
