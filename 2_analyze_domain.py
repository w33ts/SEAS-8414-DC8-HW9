"""
End-to-end analyzer:
1) Load MOJO model (model/DGA_Leader.zip)
2) Compute features for an input domain (length, entropy)
3) Predict DGA vs legit
4) Explain with SHAP (KernelExplainer)
5) Summarize explanation and generate a prescriptive playbook via Gemini

Usage:
  python 2_analyze_domain.py <domain>

Environment:
  GOOGLE_API_KEY must be set for playbook generation
"""

import asyncio
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import aiohttp
import h2o
import pandas as pd
import shap
import numpy as np


def compute_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts: Dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    length = float(len(s))
    return -sum((c / length) * math.log((c / length), 2) for c in counts.values())


def locate_mojo() -> str:
    candidates = ["model/DGA_Leader.zip", "./model/DGA_Leader.zip"]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "MOJO not found. Run 1_train_and_export.py first to create model/DGA_Leader.zip"
    )


def ensure_h2o() -> None:
    try:
        # If already running, this is a no-op
        h2o.init()
    except Exception:
        # Try once more with default init
        h2o.init()


def predict_with_mojo(mojo_path: str, domain: str) -> Tuple[pd.DataFrame, float]:
    ensure_h2o()

    length = len(domain)
    entropy = compute_entropy(domain)
    df = pd.DataFrame([{"length": length, "entropy": entropy}])
    h2o_frame = h2o.H2OFrame(df)

    model = h2o.import_mojo(mojo_path)
    preds = model.predict(h2o_frame)
    preds_df = preds.as_data_frame()

    # H2O binomial MOJO typically outputs columns: predict, p0, p1 (or class names)
    # We assume class 'dga' exists; otherwise fallback to the last probability column.
    dga_prob = None
    for col in ["dga", "p1"]:
        if col in preds_df.columns:
            dga_prob = float(preds_df[col].iloc[0])
            break
    if dga_prob is None:
        # Fallback to last numeric column
        prob_cols = [c for c in preds_df.columns if c != "predict"]
        if prob_cols:
            dga_prob = float(preds_df[prob_cols[-1]].iloc[0])
        else:
            dga_prob = 0.0

    return df, dga_prob


def compute_shap(df_row: pd.DataFrame, background: pd.DataFrame) -> Tuple[List[float], float, List[str]]:
    # For KernelExplainer we need a prediction function returning probability of positive class
    # We will reuse the already loaded local model pipeline via H2OFrame predictions.
    # To avoid reloading the MOJO in this function, we keep H2O global and use a closure.

    ensure_h2o()

    # Build a simple prediction function over pandas data
    from functools import partial

    def predict_proba_pandas(mojo_path: str, data_like: pd.DataFrame) -> np.ndarray:
        model = h2o.import_mojo(mojo_path)
        h2o_frame = h2o.H2OFrame(pd.DataFrame(data_like, columns=["length", "entropy"]))
        preds = model.predict(h2o_frame).as_data_frame()
        if "dga" in preds.columns:
            return np.asarray(preds["dga"].values, dtype=float)
        if "p1" in preds.columns:
            return np.asarray(preds["p1"].values, dtype=float)
        # Fallback: last column as prob
        prob_cols = [c for c in preds.columns if c != "predict"]
        return np.asarray(preds[prob_cols[-1]].values, dtype=float)

    mojo_path = locate_mojo()
    f = partial(predict_proba_pandas, mojo_path)

    explainer = shap.KernelExplainer(f, background)
    shap_values = explainer.shap_values(df_row)
    expected_value = explainer.expected_value

    feature_names = ["length", "entropy"]
    # shap_values comes back as array-like aligned to feature order
    shap_list = [float(x) for x in list(shap_values[0])]
    return shap_list, float(expected_value), feature_names


async def generate_playbook(xai_findings: str, api_key: str) -> str:
    prompt = f"""
As a SOC Manager, create a step-by-step incident response playbook for a Tier 1 analyst.
Base only on the alert and explanation below. Be prescriptive. 3-5 numbered steps.

Alert & Explanation:
{xai_findings}
"""

    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash-latest:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
        ]
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                result = await response.json()
                if response.status != 200:
                    return f"Error: API {response.status}. {json.dumps(result)}"
                if result.get("candidates"):
                    return result["candidates"][0]["content"]["parts"][0]["text"]
                return "Error: Empty response from model."
    except Exception as e:
        return f"Error calling Gemini: {e}"


def shap_summary_text(domain: str, dga_prob: float, shap_vals: List[float], expected_value: float, feature_names: List[str], instance: pd.Series) -> str:
    label = "DGA" if dga_prob >= 0.5 else "Legit"
    lines = [
        f"Alert: Domain classification {label} with probability {dga_prob:.2%}.",
        f"Domain: {domain}",
        f"Features: length={instance['length']}, entropy={instance['entropy']:.4f}",
        "Top contributing factors (SHAP):",
    ]
    # Pair features with shap values and sort by absolute impact
    pairs = list(zip(feature_names, shap_vals))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    for name, val in pairs:
        direction = "increased" if val >= 0 else "decreased"
        lines.append(f"- {name} {direction} the DGA likelihood by {abs(val):.4f}")
    lines.append(f"Expected value baseline (log-odds/prob domain): {expected_value}")
    return "\n".join(lines)


async def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python 2_analyze_domain.py <domain>")
        sys.exit(1)

    domain = sys.argv[1].strip()
    mojo_path = locate_mojo()

    # Predict
    instance_df, dga_prob = predict_with_mojo(mojo_path, domain)

    # Background for SHAP: a few synthetic points around typical ranges
    background = pd.DataFrame([
        {"length": 8, "entropy": compute_entropy("google.com")},
        {"length": 10, "entropy": compute_entropy("github.com")},
        {"length": 20, "entropy": compute_entropy("x1a9q0z7l2b3n4.com")},
        {"length": 24, "entropy": compute_entropy("p9q8w7e6r5t4y3u2i1o.com")},
    ])

    shap_vals, expected_value, feature_names = compute_shap(instance_df, background)

    summary = shap_summary_text(
        domain,
        dga_prob,
        shap_vals,
        expected_value,
        feature_names,
        instance_df.iloc[0],
    )

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set. Skipping playbook generation.")
        print("--- Explanation Summary ---")
        print(summary)
        print("--- Prediction ---")
        print(json.dumps({"domain": domain, "dga_probability": dga_prob}, indent=2))
        return

    print("--- Explanation Summary ---")
    print(summary)
    print("\n--- AI-Generated Playbook ---")
    playbook = await generate_playbook(summary, api_key)
    print(playbook)

    # Clean shutdown
    try:
        h2o.shutdown(prompt=False)
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())


