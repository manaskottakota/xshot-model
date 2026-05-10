"""FastAPI server: structured shot payload → calibrated xShot inference + static SPA."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, cast

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from xshot.datasets import PRIOR_LAST_N_FG_COL
from xshot.viz.inference_row import (
    infer_shot_zone_basic,
    inference_X_from_payload,
    resolve_model_path,
)


def _features_mode() -> Literal["core", "core+advanced"]:
    m = os.environ.get("XSHOT_FEATURES", "core+advanced").strip().lower()
    if m == "core" or m == "core+advanced":
        return cast(Literal["core", "core+advanced"], m)
    raise RuntimeError(f"Invalid XSHOT_FEATURES={m!r} (expect core | core+advanced)")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WEB_DIST = PROJECT_ROOT / "webapp" / "dist"


class ShotFeaturesPayload(BaseModel):
    loc_x_ft: float = Field(..., ge=-25, le=25, description="Horizontal feet from hoop")
    loc_y_ft: float = Field(..., ge=0.25, le=46, description="Feet court-ward along attack direction")
    is_three: bool = False

    shot_archetype: str = Field(
        default="jumper",
        description="layup dunk hook fadeaway jumper pullup stepback floater",
    )

    shot_zone_basic_override: str | None = None
    shot_distance_ft: float | None = Field(default=None, ge=0)
    score_diff: int = Field(default=0, ge=-60, le=60)
    period: int = Field(default=4, ge=1, le=8)
    minutes_remaining: float = Field(default=6.5, ge=0, le=13)
    seconds_remaining: float = Field(default=0, ge=0, le=59.9)
    shooting_team_home: bool | None = Field(default=None)
    is_playoffs: bool = Field(default=False)

    shot_clock_known: bool = False
    shot_clock_seconds: float | None = Field(default=None, ge=0, le=24)

    player_profile: str = Field(default="league_average")

    defender_distance_ft: float = Field(default=4.0, ge=0.8, le=35)
    defender_contest_azimuth_deg: float = Field(default=0, ge=-90, le=90)
    dribbles_before_shot: float = Field(default=1.0, ge=0, le=20)
    touch_time_sec: float = Field(default=1.5, ge=0.1, le=22)
    time_since_catch: float = Field(default=0.8, ge=0, le=22)
    distance_traveled_before_shot: float = Field(default=1.2, ge=0, le=120)
    rest_days_since_prev_game: float = Field(default=2.0, ge=0, le=360)
    is_back_to_back: bool = False


MODEL_PATH = resolve_model_path(
    Path(p).resolve()
    if (p := os.environ.get("XSHOT_MODEL_PATH"))
    else None
)
FEATURE_MODE = _features_mode()
_CLF = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _CLF  # noqa: PLW0603 — module singleton intentional
    mp = MODEL_PATH
    if mp is not None:
        try:
            _CLF = joblib.load(mp)
        except Exception as exc:  # pragma: no cover — runtime IO
            _CLF = None
            app.state.load_error = repr(exc)
    else:
        app.state.load_error = "no artifact"
    yield


app = FastAPI(title="xShot viz", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/meta")
async def meta() -> dict[str, Any]:
    return {
        "model_path": str(MODEL_PATH) if MODEL_PATH else None,
        "model_loaded": bool(_CLF),
        "features_mode": FEATURE_MODE,
        "prior_fg_col": PRIOR_LAST_N_FG_COL,
        "player_profiles": sorted(
            ("league_average", "elite_spot_up", "rim_finisher_big", "combo_guard")
        ),
    }


@app.post("/api/predict")
async def predict(body: ShotFeaturesPayload) -> dict[str, Any]:
    if MODEL_PATH is None or _CLF is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artifact missing. Train with `python scripts/train.py` or set "
                "`XSHOT_MODEL_PATH`."
            ),
        )

    payload = body.model_dump()

    try:
        x = inference_X_from_payload(payload, features=FEATURE_MODE)
        prob_arr = _CLF.predict_proba(x)[:, 1]
        p = float(prob_arr.flat[0])
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prediction failed ({e!r}); ensure `XSHOT_FEATURES` matches training "
                f"(current `{FEATURE_MODE}`)."
            ),
        ) from e

    if not (0 <= p <= 1):
        raise HTTPException(status_code=500, detail="Malformed probability.")

    if payload.get("shot_zone_basic_override"):
        zone_label = str(payload["shot_zone_basic_override"])
    else:
        zone_label = infer_shot_zone_basic(
            float(payload["loc_x_ft"]),
            float(payload["loc_y_ft"]),
            bool(payload.get("is_three")),
        )

    return {
        "probability": round(p, 5),
        "shot_zone_infer": zone_label,
        "artifact": MODEL_PATH.name if MODEL_PATH else None,
        "features_mode": FEATURE_MODE,
    }


_if_index = WEB_DIST / "index.html"
if WEB_DIST.exists() and _if_index.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="spa")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(prog="xshot-viz")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)

    uvicorn.run(
        "xshot.viz.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
