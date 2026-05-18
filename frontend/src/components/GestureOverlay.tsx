type Landmark = { x: number; y: number; z: number };

type HandPayload = {
  handedness?: string;
  landmarks?: Landmark[];
};

type Props = {
  landmarks: unknown;
};

export function GestureOverlay({ landmarks }: Props) {
  if (!Array.isArray(landmarks) || landmarks.length === 0) return null;
  const hand = landmarks[0] as HandPayload;
  const pts = hand.landmarks;
  if (!pts || pts.length !== 21) return null;

  const connections: [number, number][] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [0, 17],
  ];

  return (
    <svg
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        transform: "scaleX(-1)",
      }}
    >
      {connections.map(([a, b], i) => {
        const pa = pts[a];
        const pb = pts[b];
        return (
          <line
            key={i}
            x1={pa.x}
            y1={pa.y}
            x2={pb.x}
            y2={pb.y}
            stroke="#38bdf8"
            strokeWidth={0.004}
          />
        );
      })}
      {pts.map((p, i) => (
        <circle key={i} cx={p.x} cy={p.y} r={0.012} fill="#fbbf24" />
      ))}
    </svg>
  );
}
