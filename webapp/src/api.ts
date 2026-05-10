export type FeaturesMode = "core" | "core+advanced";

export type MetaResponse = {
  model_path: string | null;
  model_loaded: boolean;
  features_mode: FeaturesMode;
  prior_fg_col: string;
  player_profiles: string[];
};

export type PredictResponse = {
  probability: number;
  shot_zone_infer: string;
  artifact: string | null;
  features_mode: FeaturesMode;
};

export type PredictBody = Record<string, unknown>;

export async function fetchMeta(): Promise<MetaResponse> {
  const r = await fetch("/api/meta");
  if (!r.ok) throw new Error(`meta failed: ${r.status}`);
  return (await r.json()) as MetaResponse;
}

export async function predictShot(body: PredictBody): Promise<PredictResponse> {
  const r = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const j = (await r.json()) as PredictResponse | { detail: unknown };
  if (!r.ok) {
    const msg =
      typeof (j as { detail?: unknown }).detail === "string"
        ? (j as { detail: string }).detail
        : JSON.stringify((j as { detail?: unknown }).detail ?? j);
    throw new Error(msg);
  }
  return j as PredictResponse;
}
