import { useCallback, useEffect, useRef, useState } from "react";

type Props = {
  locX: number;
  locY: number;
  probability: number | null;
  onChange: (lx: number, ly: number) => void;
};

const X_MIN = -24.75;
const X_MAX = 24.75;
const Y_MIN = 0.35;
const Y_MAX = 45.85;

/** Red → yellow → green hue sweep keyed by calibrated probability [0,1]. */
function probToColor(p: number): { fill: string; glow: string } {
  const t = Math.min(1, Math.max(0, p));
  let hueMid: number;
  if (t <= 0.5) hueMid = 4 + ((48 - 4) * t) / 0.5;
  else hueMid = 48 + ((132 - 48) * (t - 0.5)) / 0.5;
  const fill = `hsl(${hueMid.toFixed(2)}, 96%, ${44 + t * 9}%)`;
  const glow = `hsla(${hueMid.toFixed(2)}, 96%, ${58 + t * 6}%, ${0.25 + t * 0.5})`;
  return { fill, glow };
}

export function HalfCourt({ locX, locY, probability, onChange }: Props) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const p = probability ?? 0;
  const { fill: shotFill, glow } = probToColor(probability === null ? 0.25 : p);
  const outerR = probability === null ? 1.06 : 0.88 + probability * 0.7;

  const toCourtCoords = useCallback((clientX: number, clientY: number) => {
    const svg = svgRef.current;
    if (!svg) return null;
    const pt = svg.createSVGPoint();
    pt.x = clientX;
    pt.y = clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return null;
    const inv = pt.matrixTransform(ctm.inverse());
    return {
      lx: Math.min(X_MAX, Math.max(X_MIN, inv.x)),
      ly: Math.min(Y_MAX, Math.max(Y_MIN, inv.y)),
    };
  }, []);

  const dispatchMove = useCallback(
    (e: React.PointerEvent | PointerEvent) => {
      const coords = toCourtCoords(e.clientX, e.clientY);
      if (coords) onChange(coords.lx, coords.ly);
    },
    [onChange, toCourtCoords],
  );

  useEffect(() => {
    const cancel = () => setDragging(false);
    window.addEventListener("mouseup", cancel);
    window.addEventListener("touchend", cancel);
    return () => {
      window.removeEventListener("mouseup", cancel);
      window.removeEventListener("touchend", cancel);
    };
  }, []);

  useEffect(() => {
    if (!dragging) return undefined;
    const move = (e: PointerEvent) => dispatchMove(e as unknown as React.PointerEvent);
    window.addEventListener("pointermove", move);
    return () => window.removeEventListener("pointermove", move);
  }, [dragging, dispatchMove]);

  return (
    <div className="court-wrap">
      <svg ref={svgRef} role="img" aria-label="Half court shot chart" viewBox="-26 -4 52 50">
        <defs>
          <radialGradient id="shotPulseCore">
            <stop offset="35%" stopColor={shotFill} stopOpacity={1} />
            <stop offset="100%" stopColor={shotFill} stopOpacity={0} />
          </radialGradient>
          <filter id="courtGlowBlur">
            <feGaussianBlur
              stdDeviation={
                probability === null ? "0.95" : `${0.82 + probability * 1.9}`
              }
            />
          </filter>
        </defs>

        <rect x="-26" y="-4" width="52" height="50" rx="3" ry="3" fill="#14181e" />
        <rect x="-25.1" y="-0.95" width="50.2" height="46.6" rx="2.05" ry="2.05" fill="#161a23" />

        <g opacity={0.11} stroke="rgba(236,246,255,0.42)" strokeWidth={0.04}>
          {[...Array(13)].map((_, i) => (
            <line key={`gv${String(i)}`} x1={-25 + i * 5} x2={-25 + i * 5} y1="0.2" y2="46" />
          ))}
        </g>

        {/* Court markings */}
        <g fill="none" stroke="#cdd6e3" strokeWidth={0.12} strokeLinecap="round">
          <rect x={-21.94} y={0.08} width="43.88" height="43.94" />

          {/* Paint */}
          <rect x={-7.95} y={0} width="15.9" height="17.92" />

          {/* Free throw */}
          <line x1="-7.92" x2="7.92" y1={16.82} y2={16.82} />

          {/* Restricted */}
          <path d="M -4.08 0.25 A 4.08 4.08 0 0 0 4.08 0.25" />

          <defs>
            <clipPath id="arcClip">
              <rect x="-50" y="-4" width="100" height="55" />
            </clipPath>
          </defs>
          <circle clipPath="url(#arcClip)" cx={0} cy={0} r={23.88} />

          {/* Rim + backboard */}
          <circle cx={0} cy={-0.35} r={0.78} fill="rgba(246,251,255,0.96)" stroke="none" />
          <line x1="-7.1" x2="7.1" y={-1.55} y2={-1.55} strokeWidth={0.42} />

          {/* Midcourt dashed guide */}
          <line
            x1="-21.4"
            y1={42.92}
            x2="21.4"
            y2={42.92}
            stroke="rgba(229,239,251,0.18)"
            strokeWidth={0.08}
            strokeDasharray="0.92 1.06"
          />
        </g>

        {/* Shot glow + pulse (center at rim-oriented coordinates) */}
        <g pointerEvents="none">
          <g transform={`translate(${locX},${locY})`} filter={probability !== null ? "url(#courtGlowBlur)" : undefined}>
            {!dragging ? (
              <g className="halo-motion">
                <circle r={outerR + 1.95} fill={glow} opacity={probability === null ? 0 : 0.7} />
                <circle r={outerR + 0.92} fill="none" opacity={probability === null ? 0 : 0.82} stroke={shotFill} strokeWidth={0.07} />
              </g>
            ) : (
              <circle r={outerR + 0.92} fill={glow} opacity={probability === null ? 0 : 0.92} />
            )}
            <circle
              cx={0}
              cy={0}
              r={0.94}
              fill="url(#shotPulseCore)"
              stroke={probability === null ? "rgba(255,255,255,0.3)" : "rgba(253,246,237,0.55)"}
              strokeWidth={0.06}
            />
          </g>
        </g>

        {/* Pointer capture */}
        <rect
          x="-26"
          y="-1"
          width="52"
          height="46.5"
          fill="transparent"
          style={{ cursor: "crosshair" }}
          onPointerDown={(e) => {
            (e.target as HTMLElement).setPointerCapture(e.pointerId);
            setDragging(true);
            dispatchMove(e);
          }}
          onPointerUp={(e) => {
            dispatchMove(e);
            try {
              (e.target as HTMLElement).releasePointerCapture(e.pointerId);
            } catch {
              //
            }
            setDragging(false);
          }}
        />
      </svg>

      <style>{`
        @keyframes haloDrift {
          0% {
            transform: scale(0.94);
            opacity: 0.55;
          }
          52% {
            transform: scale(1.06);
            opacity: 1;
          }
          100% {
            transform: scale(0.94);
            opacity: 0.55;
          }
        }
        .halo-motion {
          transform-origin: center;
          animation: haloDrift 2.15s ease-in-out infinite;
        }
        .court-wrap {
          position: relative;
          width: min(760px, 100%);
          aspect-ratio: 52 / 50;
          border-radius: 16px;
          border: 1px solid rgba(198, 213, 230, 0.12);
          box-shadow:
            inset 0 1px rgba(255, 255, 255, 0.03),
            0 10px 50px rgba(0, 0, 0, 0.64);
          background: radial-gradient(circle at 50% -5%, rgba(34, 45, 64, 0.35), transparent 62%);
          overflow: hidden;
        }
        .court-wrap svg {
          width: 100%;
          height: 100%;
          display: block;
        }
      `}</style>
    </div>
  );
}
