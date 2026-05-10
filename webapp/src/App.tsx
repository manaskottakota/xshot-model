import { useEffect, useMemo, useRef, useState } from "react";
import { HalfCourt } from "./components/HalfCourt";
import type { FeaturesMode } from "./api";
import { fetchMeta, predictShot } from "./api";

type HomeTri = "unset" | "home" | "away";

const ZONE_CHOICES = [
  "",
  "Restricted Area",
  "In The Paint (Non-RA)",
  "Mid-Range",
  "Left Corner 3",
  "Right Corner 3",
  "Above the Break 3",
  "Backcourt",
];

const ARCHETYPES = [
  "layup",
  "dunk",
  "floater",
  "hook",
  "fadeaway",
  "pullup",
  "stepback",
  "jumper",
] as const;

function triToHome(flag: HomeTri): boolean | null | undefined {
  if (flag === "unset") return undefined;
  if (flag === "home") return true;
  return false;
}

export function App() {
  const [meta, setMeta] = useState<{
    loaded: boolean;
    model_loaded: boolean;
    features_mode: FeaturesMode | null;
    player_profiles: string[];
    detail?: string;
  }>({
    loaded: false,
    model_loaded: false,
    features_mode: null,
    player_profiles: [],
  });

  const [locX, setLocX] = useState(-18.75);
  const [locY, setLocY] = useState(22.05);
  const [probability, setProbability] = useState<number | null>(null);
  const [zoneInfer, setZoneInfer] = useState<string>("—");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [isThree, setIsThree] = useState(true);
  const [archetype, setArchetype] = useState<(typeof ARCHETYPES)[number]>("jumper");
  const [zoneOverride, setZoneOverride] = useState("");
  const [period, setPeriod] = useState(4);
  const [mins, setMins] = useState(6);
  const [secs, setSecs] = useState(12);
  const [scoreDiff, setScoreDiff] = useState(0);
  const [home, setHome] = useState<HomeTri>("unset");
  const [playoffs, setPlayoffs] = useState(false);

  const [scKnown, setScKnown] = useState(false);
  const [scSec, setScSec] = useState(8.8);

  const [playerProfile, setPlayerProfile] = useState("league_average");

  const [defFt, setDefFt] = useState(4.05);
  const [defPsi, setDefPsi] = useState(12);
  const [dribs, setDribs] = useState(1.1);
  const [touchSec, setTouchSec] = useState(1.65);
  const [sinceCatch, setSinceCatch] = useState(0.85);
  const [travel, setTravel] = useState(1.15);
  const [restDays, setRestDays] = useState(2.0);
  const [btb, setBtb] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const m = await fetchMeta();
        setMeta({
          loaded: true,
          model_loaded: m.model_loaded,
          features_mode: m.features_mode,
          player_profiles: m.player_profiles,
        });
      } catch (e) {
        setMeta((s) => ({
          ...s,
          loaded: true,
          detail: e instanceof Error ? e.message : String(e),
        }));
      }
    })();
  }, []);

  const distHint = useMemo(() => Math.hypot(locX, locY), [locX, locY]);

  const advanced = meta.features_mode === "core+advanced";

  const execPredict = useRef<() => Promise<void>>(async () => {});

  async function handleSubmit() {
    setError(null);
    setLoading(true);
    setProbability(null);
    try {
      const payload: Record<string, unknown> = {
        loc_x_ft: Number(locX.toFixed(6)),
        loc_y_ft: Number(locY.toFixed(6)),
        is_three: isThree,
        shot_archetype: archetype,
        shot_zone_basic_override: zoneOverride || null,
        shot_distance_ft: Number(distHint.toFixed(6)),
        period,
        minutes_remaining: mins,
        seconds_remaining: secs,
        score_diff: scoreDiff,
        shooting_team_home: triToHome(home),
        is_playoffs: playoffs,
        shot_clock_known: scKnown,
        shot_clock_seconds: scKnown ? scSec : null,
        player_profile: playerProfile,
        defender_distance_ft: defFt,
        defender_contest_azimuth_deg: defPsi,
        dribbles_before_shot: dribs,
        touch_time_sec: touchSec,
        time_since_catch: sinceCatch,
        distance_traveled_before_shot: travel,
        rest_days_since_prev_game: restDays,
        is_back_to_back: btb,
      };
      const res = await predictShot(payload);
      setProbability(res.probability);
      setZoneInfer(res.shot_zone_infer);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  execPredict.current = handleSubmit;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "enter") {
        e.preventDefault();
        void execPredict.current();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const badge = meta.model_loaded ? "model online" : "model offline";

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">xS</div>
          <div>
            <div className="brand-title">xShot Lab · v1</div>
            <div className="brand-subtle">Structured halfcourt explorer — inference-only</div>
          </div>
        </div>
        <div className={`pill ${meta.model_loaded ? "pill-live" : "pill-idle"}`}>{badge}</div>
      </header>

      {!meta.loaded ? (
        <p className="muted">Initializing…</p>
      ) : (
        <>
          {!meta.model_loaded ? (
            <div className="banner">
              Attach a calibrated pipeline first: train locally, then rerun with artifact under{" "}
              <code>/artifacts/run_default/</code>. Adjust <code>XSHOT_MODEL_PATH</code> /
              <code>XSHOT_FEATURES</code> env if versions diverge.
            </div>
          ) : null}
          {meta.detail ? <div className="banner-warn">{meta.detail}</div> : null}
          <div className="grid-main">
            <section className="court-card">
              <div className="court-head">
                <div>
                  <h2>Court projection</h2>
                  <p className="muted small">
                    Drag on the parquet to move the shooter. Rim axis at the top baseline; arcs open
                    toward midcourt ({distHint.toFixed(1)} ft geometric distance).
                  </p>
                </div>
              </div>
              <HalfCourt
                locX={locX}
                locY={locY}
                probability={probability}
                onChange={(lx, ly) => {
                  setLocX(Number(lx.toFixed(6)));
                  setLocY(Number(ly.toFixed(6)));
                  setProbability(null);
                  setZoneInfer("—");
                }}
              />

              <div className={`readout ${probability !== null ? "live" : ""}`}>
                <div className="readout-row">
                  <span className="muted">Calibrated FG probability</span>
                  <strong className="prob-monospace readout-score">
                    {probability === null ? "—" : `${(probability * 100).toFixed(2)}%`}
                  </strong>
                </div>
                <div className="infer-row">
                  <span className="muted">Shot zone heuristic</span>
                  <span className="mono">{zoneInfer}</span>
                </div>
              </div>
            </section>

            <aside className="controls">
              <div className="panel">
                <h3 className="ctrl-title">Shot type</h3>
                <div className="field-row">
                  <label htmlFor="arche">
                    Shot archetype
                    <select
                      id="arche"
                      value={archetype}
                      onChange={(e) => setArchetype(e.target.value as typeof archetype)}
                    >
                      {ARCHETYPES.map((a) => (
                        <option key={a} value={a}>
                          {a.replaceAll("_", " ")}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="chk">
                    <input
                      type="checkbox"
                      checked={isThree}
                      onChange={(e) => {
                        setIsThree(e.target.checked);
                        setProbability(null);
                      }}
                    />
                    Three-point FGA
                  </label>
                </div>
                <label htmlFor="zone">
                  Override zone categorical
                  <select id="zone" value={zoneOverride} onChange={(e) => setZoneOverride(e.target.value)}>
                    {ZONE_CHOICES.map((z) => (
                      <option key={`z:${z.length ? z : "auto"}`} value={z}>
                        {z === "" ? "Auto-from-spot geometry" : z}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="panel">
                <h3 className="ctrl-title">Game situation</h3>
                <div className="field-split">
                  <label>
                    Quarter
                    <select value={period} onChange={(e) => setPeriod(Number(e.target.value))}>
                      {[1, 2, 3, 4].map((p) => (
                        <option key={p} value={p}>{`Quarter ${p}`}</option>
                      ))}
                      {[5, 6].map((p) => (
                        <option key={p} value={p}>{`OT ${String(p - 4)}`}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Clock (min:s)
                    <div className="inline">
                      <input
                        aria-label="Minutes remaining"
                        type="number"
                        min={0}
                        step={1}
                        value={mins}
                        onChange={(e) => setMins(Number(e.target.value))}
                      />
                      <span>:</span>
                      <input
                        aria-label="Seconds remaining"
                        type="number"
                        min={0}
                        max={59}
                        step={1}
                        value={secs}
                        onChange={(e) => setSecs(Number(e.target.value))}
                      />
                    </div>
                  </label>
                </div>
                <label>
                  Shooting team margin (+ ahead)
                  <input
                    type="number"
                    min={-55}
                    max={55}
                    value={scoreDiff}
                    onChange={(e) => setScoreDiff(Number(e.target.value))}
                  />
                </label>

                <div className="radio-row muted">
                  <span>Shooting-team homecourt</span>
                  <label>
                    <input type="radio" name="hh" checked={home === "unset"} onChange={() => setHome("unset")} />
                    omit
                  </label>
                  <label>
                    <input type="radio" name="hh" checked={home === "home"} onChange={() => setHome("home")} />
                    home
                  </label>
                  <label>
                    <input type="radio" name="hh" checked={home === "away"} onChange={() => setHome("away")} />
                    away
                  </label>
                </div>

                <label className="chk muted">
                  <input type="checkbox" checked={playoffs} onChange={(e) => setPlayoffs(e.target.checked)} />
                  Playoffs
                </label>
              </div>

              <div className="panel">
                <h3 className="ctrl-title">Shot clock</h3>
                <label className="chk">
                  <input type="checkbox" checked={scKnown} onChange={(e) => setScKnown(e.target.checked)} />
                  Shot-clock value known / supplied
                </label>
                {scKnown ? (
                  <label>
                    Seconds
                    <input
                      type="number"
                      min={0}
                      max={24}
                      step={0.1}
                      value={scSec}
                      onChange={(e) => setScSec(Number(e.target.value))}
                    />
                  </label>
                ) : (
                  <p className="tiny muted">
                    Sentinel model path sets clock unknown (<code>-1</code>), matching training data gaps.
                  </p>
                )}
              </div>

              <div className="panel">
                <h3 className="ctrl-title">Shooter preset</h3>
                <p className="tiny muted">
                  Controls rolling-style priors (not NLP). Swap profile to emulate archetypes offline.
                </p>
                <select value={playerProfile} onChange={(e) => setPlayerProfile(e.target.value)}>
                  {meta.player_profiles.map((n) => (
                    <option key={n} value={n}>
                      {n.replace(/_/g, " ")}
                    </option>
                  ))}
                </select>
              </div>

              <div className={`tracking-panel panel ${advanced ? "" : "is-inert"}`}>
                <div className="panel-head-inline">
                  <h3 className="ctrl-title">Tracking / contextual</h3>
                  {!advanced ? <span className="muted tiny">inactive for pure core manifests</span> : null}
                </div>

                <label>
                  Defender distance · ft
                  <input
                    type="range"
                    min={0.95}
                    max={18}
                    step={0.05}
                    value={defFt}
                    disabled={!advanced}
                    onChange={(e) => setDefFt(Number(e.target.value))}
                  />
                  <div className="range-read prob-monospace">{defFt.toFixed(2)}′</div>
                </label>

                <label>
                  Contest azimuth (−sideline/+slot)
                  <input
                    type="range"
                    min={-80}
                    max={80}
                    step={1}
                    value={defPsi}
                    disabled={!advanced}
                    onChange={(e) => setDefPsi(Number(e.target.value))}
                  />
                  <div className="range-read prob-monospace">{defPsi.toFixed()}° off radial</div>
                </label>

                <label>
                  Dribbles
                  <input
                    type="range"
                    min={0}
                    max={12}
                    step={0.05}
                    value={dribs}
                    disabled={!advanced}
                    onChange={(e) => setDribs(Number(e.target.value))}
                  />
                  <div className="range-read">{dribs.toFixed(2)}</div>
                </label>

                <label>
                  Touch seconds
                  <input
                    type="range"
                    min={0.3}
                    max={12}
                    step={0.05}
                    value={touchSec}
                    disabled={!advanced}
                    onChange={(e) => setTouchSec(Number(e.target.value))}
                  />
                  <div className="range-read">{touchSec.toFixed(2)} s</div>
                </label>

                <label>
                  Time since catch
                  <input
                    type="range"
                    min={0}
                    max={8}
                    step={0.05}
                    value={sinceCatch}
                    disabled={!advanced}
                    onChange={(e) => setSinceCatch(Number(e.target.value))}
                  />
                  <div className="range-read">{sinceCatch.toFixed(2)} s</div>
                </label>

                <label>
                  Pre-shot travel ft
                  <input
                    type="range"
                    min={0}
                    max={30}
                    step={0.05}
                    value={travel}
                    disabled={!advanced}
                    onChange={(e) => setTravel(Number(e.target.value))}
                  />
                  <div className="range-read">{travel.toFixed(2)}′</div>
                </label>

                <label>
                  Rest days vs previous game
                  <input
                    type="range"
                    min={0}
                    max={9}
                    step={0.05}
                    value={restDays}
                    disabled={!advanced}
                    onChange={(e) => setRestDays(Number(e.target.value))}
                  />
                  <div className="range-read">{restDays.toFixed(2)} d</div>
                </label>
                <label className="chk muted">
                  <input type="checkbox" disabled={!advanced} checked={btb} onChange={(e) => setBtb(e.target.checked)} />
                  Back-to-back
                </label>
              </div>

              <div className="stack-actions">
                {error ? <div className="banner-warn">{error}</div> : null}
                <button type="button" className="primary" disabled={loading || !meta.model_loaded} onClick={() => void handleSubmit()}>
                  {loading ? "Evaluating probability…" : "Run calibrated model"}
                </button>
              </div>
            </aside>
          </div>
          <footer className="foot muted">
            <span>
              <kbd className="kbd">⌘ / Ctrl</kbd> + <kbd className="kbd">Enter</kbd> → fire inference
            </span>
          </footer>
        </>
      )}

      <style>{`
        .mono {
          font-family: ui-monospace, monospace;
          font-size: 0.85rem;
        }
        code {
          padding: 0.12rem 0.35rem;
          border-radius: 5px;
          background: rgba(255, 255, 255, 0.07);
          font-size: 0.74rem;
        }
        .shell {
          padding: clamp(22px, 4vw, 44px);
          max-width: 1260px;
          margin-inline: auto;
        }
        .topbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          margin-bottom: 22px;
        }
        .brand {
          display: flex;
          gap: 12px;
          align-items: center;
        }
        .brand-mark {
          width: 44px;
          height: 44px;
          border-radius: 12px;
          background: radial-gradient(circle at 30% -10%, rgba(94, 150, 255, 0.35), transparent 72%),
            linear-gradient(130deg, #1c2433, #0e111a);
          border: 1px solid rgba(190, 210, 238, 0.15);
          display: grid;
          place-items: center;
          font-weight: 700;
          font-size: 0.85rem;
        }
        .brand-title {
          font-size: clamp(18px, 2.4vw, 22px);
          font-weight: 650;
        }
        .brand-subtle {
          font-size: 0.74rem;
          color: rgba(228, 235, 245, 0.55);
        }
        .pill {
          padding: 0.42rem 0.92rem;
          border-radius: 999px;
          font-size: 0.73rem;
          letter-spacing: 0.045em;
          text-transform: uppercase;
          border: 1px solid rgba(226, 234, 244, 0.16);
          background: rgba(14, 16, 20, 0.35);
          color: rgba(232, 240, 250, 0.65);
          transition: opacity 0.2s ease, border-color 0.2s ease;
        }
        .pill-live {
          border-color: rgba(112, 196, 120, 0.52);
          color: rgba(188, 250, 200, 0.85);
          background: radial-gradient(circle at 30% -10%, rgba(52, 120, 64, 0.28), transparent 74%);
          box-shadow: 0 0 18px rgba(120, 200, 120, 0.15);
        }
        .pill-idle {
          border-color: rgba(220, 150, 90, 0.42);
          color: rgba(255, 200, 150, 0.92);
          background: rgba(80, 40, 22, 0.18);
        }
        .muted {
          color: rgba(230, 236, 246, 0.57);
          font-weight: 400;
          font-style: normal;
        }
        .small {
          font-size: 0.78rem;
        }
        .tiny {
          margin: 0 0 0.45rem;
          font-size: 0.71rem;
        }
        .banner,
        .banner-warn {
          border-radius: 12px;
          padding: 0.75rem 0.92rem;
          margin-bottom: 16px;
          font-size: 0.82rem;
        }
        .banner {
          background: rgba(64, 86, 120, 0.25);
          border: 1px solid rgba(150, 180, 220, 0.18);
          color: rgba(233, 240, 250, 0.78);
          line-height: 1.4;
        }
        .banner-warn {
          background: rgba(150, 50, 50, 0.18);
          border: 1px solid rgba(255, 155, 120, 0.28);
          color: rgba(255, 226, 220, 0.93);
          line-height: 1.4;
          white-space: pre-wrap;
        }
        .grid-main {
          display: grid;
          grid-template-columns: minmax(0, 7fr) minmax(286px, 4fr);
          gap: clamp(18px, 3vw, 28px);
        }
        @media (max-width: 992px) {
          .grid-main {
            grid-template-columns: minmax(0, 1fr);
          }
        }
        .court-card {
          display: grid;
          gap: 14px;
        }
        .court-head h2 {
          margin: 0 0 0.38rem;
          font-size: 1.06rem;
        }
        .readout {
          border-radius: 16px;
          border: 1px solid rgba(210, 220, 238, 0.12);
          padding: clamp(13px, 2.4vw, 18px) clamp(16px, 3vw, 22px);
          background: radial-gradient(circle at 10% -20%, rgba(120, 150, 200, 0.16), transparent 65%),
            rgba(26, 30, 39, 0.88);
          transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }
        .readout.live {
          border-color: rgba(130, 200, 150, 0.28);
          box-shadow: inset 0 0 56px rgba(90, 200, 120, 0.08);
        }
        .readout-row {
          display: flex;
          justify-content: space-between;
          align-items: baseline;
          gap: 12px;
        }
        .readout-score {
          font-size: clamp(20px, 3.4vw, 28px);
        }
        .infer-row {
          margin-top: 0.72rem;
          display: flex;
          justify-content: space-between;
          gap: 8px;
        }
        .controls {
          display: flex;
          flex-direction: column;
          gap: 13px;
        }
        .panel {
          backdrop-filter: blur(10px);
          background: var(--panel);
          border-radius: 16px;
          border: 1px solid rgba(226, 234, 244, 0.12);
          padding: 13px 16px;
          box-shadow:
            inset 0 1px rgba(255, 255, 255, 0.05),
            0 6px 32px rgba(0, 0, 0, 0.5);
          display: flex;
          flex-direction: column;
          gap: 0.65rem;
        }
        .tracking-panel.is-inert {
          opacity: 0.61;
          filter: saturate(0.78);
        }
        .panel-head-inline {
          display: flex;
          justify-content: space-between;
          align-items: center;
          gap: 8px;
        }
        .ctrl-title {
          margin: 0;
          font-size: 0.85rem;
          letter-spacing: 0.085em;
          text-transform: uppercase;
          opacity: 0.78;
        }
        select,
        input:not([type="checkbox"]):not([type="radio"]) {
          width: 100%;
          margin-top: 0.3rem;
          padding: 0.55rem 0.72rem;
          border-radius: 10px;
          border: 1px solid rgba(210, 220, 238, 0.15);
          background: rgba(12, 15, 20, 0.88);
          color: inherit;
          outline: none;
          transition:
            border-color 0.2s ease,
            box-shadow 0.25s ease;
        }
        select:focus-visible,
        input:not([type="checkbox"]):not([type="radio"]):focus-visible {
          border-color: rgba(154, 200, 255, 0.55);
          box-shadow: 0 0 0 2px rgba(120, 170, 255, 0.18);
        }
        label.chk {
          display: flex;
          align-items: center;
          gap: 0.62rem;
          font-size: 0.82rem;
        }
        .field-split {
          display: grid;
          gap: 0.55rem;
        }
        @media (min-width: 620px) {
          .field-split {
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.92rem;
          }
        }
        .field-row {
          display: flex;
          gap: 0.92rem;
          flex-wrap: wrap;
          justify-content: space-between;
          align-items: flex-start;
        }
        .radio-row label {
          display: inline-flex;
          gap: 0.25rem;
          align-items: center;
          margin-left: 0.45rem;
        }
        .inline {
          display: flex;
          align-items: center;
          gap: 0.4rem;
        }
        .inline input[type="number"] {
          flex: 1;
        }
        .range-read {
          font-size: 0.73rem;
          margin-top: 0.08rem;
          color: rgba(232, 240, 251, 0.65);
        }
        input[type="range"] {
          width: 100%;
          accent-color: rgb(154, 200, 255);
        }
        .stack-actions {
          display: flex;
          flex-direction: column;
          gap: 0.55rem;
        }
        .primary {
          width: 100%;
          border: none;
          border-radius: 14px;
          padding: 0.94rem;
          font-weight: 650;
          letter-spacing: 0.046em;
          text-transform: uppercase;
          cursor: pointer;
          color: #0f1218;
          background: linear-gradient(125deg, #e6f8ff 0%, #cddfff 54%, #9eeabd 164%);
          box-shadow:
            0 14px 30px rgba(140, 200, 255, 0.18),
            0 0 0 1px rgba(255, 255, 255, 0.18) inset,
            inset 0 -28px 40px rgba(140, 200, 120, 0.15);
          transition:
            opacity 0.2s ease,
            transform 0.15s ease;
        }
        .primary:disabled {
          opacity: 0.45;
          cursor: wait;
          box-shadow: none;
        }
        .primary:not(:disabled):hover {
          transform: translateY(-1px);
        }
        .foot {
          margin-top: 24px;
          font-size: 0.73rem;
        }
        .kbd {
          padding: 0.1rem 0.35rem;
          border-radius: 5px;
          border: 1px solid rgba(226, 234, 244, 0.15);
          background: rgba(18, 20, 24, 0.75);
          font-size: 0.66rem;
        }
      `}</style>
    </div>
  );
}
