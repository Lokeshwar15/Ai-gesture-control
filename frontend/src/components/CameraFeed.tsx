import { useEffect, useRef, useState } from "react";
import { useCamera } from "../hooks/useCamera";
import { useWebSocket } from "../hooks/useWebSocket";
import { GestureOverlay } from "./GestureOverlay";

export function CameraFeed() {
  const [running, setRunning] = useState(true);
  const { videoRef, canvasRef, error, captureJpegBlob } = useCamera(running);
  const { status, lastMessage, sendFrame } = useWebSocket();
  const rafRef = useRef<number>();

  useEffect(() => {
    let stopped = false;
    const loop = async () => {
      if (stopped) return;
      const blob = await captureJpegBlob();
      if (blob) sendFrame(blob);
      rafRef.current = requestAnimationFrame(loop);
    };
    if (running) rafRef.current = requestAnimationFrame(loop);
    return () => {
      stopped = true;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [running, captureJpegBlob, sendFrame]);

  const gesture = typeof lastMessage?.gesture === "string" ? lastMessage.gesture : "—";
  const confidence =
    typeof lastMessage?.confidence === "number" ? lastMessage.confidence : 0;
  const landmarks = lastMessage?.landmarks as unknown;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
        <button type="button" onClick={() => setRunning((v) => !v)}>
          {running ? "Pause camera" : "Resume camera"}
        </button>
        <span>
          WebSocket: <strong>{status}</strong>
        </span>
        <span>
          Gesture: <strong>{gesture}</strong> ({(confidence * 100).toFixed(0)}%)
        </span>
      </div>
      {error && <div style={{ color: "#f87171" }}>{error}</div>}
      <div style={{ position: "relative", width: "100%", maxWidth: 960 }}>
        <video
          ref={videoRef}
          muted
          playsInline
          autoPlay
          style={{ width: "100%", transform: "scaleX(-1)", borderRadius: 8 }}
        />
        <GestureOverlay landmarks={landmarks} />
        <canvas ref={canvasRef} style={{ display: "none" }} />
      </div>
    </div>
  );
}
