# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pandas",
# ]
# ///
"""Refresh the salary-model section of index.html from the model outputs.

Run this after ``final_model.py`` so the browser page picks up the new
coefficients, metrics, and FDR-selected tool list::

    uv run analysis/final_model.py
    uv run analysis/export_model_data.py

Only the embedded ``const DATA = {...};`` line is rewritten; the chart,
tab, and search code is left alone.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "analysis" / "out"
DEFAULT_HTML = ROOT / "index.html"


def _pct(series):
    return (series.astype(float) * 100).round(1)


def _count(series):
    return series.fillna(0).astype(int)


def build_payload(out_dir):
    metrics = json.loads((out_dir / "final_model_metrics.json").read_text(encoding="utf-8"))
    roles = pd.read_csv(out_dir / "final_model_roles.csv")
    tools = pd.read_csv(out_dir / "final_model_tools.csv")
    controls = pd.read_csv(out_dir / "final_model_controls.csv")
    curve = pd.read_csv(out_dir / "final_model_exp_curve.csv")

    def control_block(prefix):
        rows = controls[controls["term"].str.startswith(prefix)].copy()
        rows = rows.sort_values("pct_effect", ascending=False)
        return [
            {"label": str(r.term).replace(prefix, ""), "pct": float(round(r.pct_effect * 100, 1))}
            for r in rows.itertuples()
        ]

    return {
        "metrics": {
            "cv_r2": metrics["cv_r2"],
            "cv_mae": metrics["cv_mae_sgd"],
            "hold_r2": metrics["holdout_r2"],
            "hold_mae": metrics["holdout_mae_sgd"],
            "in_r2": metrics["in_sample_r2"],
            "n_features": metrics["n_features"],
            "n_rows": metrics["n_model_rows"],
            "n_roles": metrics["n_roles"],
            "n_tool_candidates": metrics.get("n_tool_candidates"),
            "n_tools_selected": metrics.get("n_tools_selected"),
        },
        "perm": [
            {"group": row["group"], "delta": round(float(row["delta_mae_sgd"]), 1)}
            for row in metrics.get("group_importance", [])
        ],
        "curve": [
            {
                "x": int(r.experience_years),
                "y": round(float(r.predicted_mid)),
                "lo": round(float(r.ci_low)),
                "hi": round(float(r.ci_high)),
            }
            for r in curve.itertuples()
        ],
        "roles": [
            {"label": str(r.role_label), "n": int(r.n_postings) if pd.notna(r.n_postings) else 0,
             "pct": float(round(r.pct_effect * 100, 1))}
            for r in roles.itertuples()
        ],
        "tools": [
            {"label": str(r.tool), "n": int(r.n_postings) if pd.notna(r.n_postings) else 0,
             "pct": float(round(r.pct_effect * 100, 1))}
            for r in tools.itertuples()
        ],
        "position": control_block("pos::"),
        "employment": control_block("emp::"),
        "district": control_block("district::"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()

    payload = build_payload(args.out)
    line = "const DATA = " + json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + ";"

    html = args.html.read_text(encoding="utf-8")
    script_at = html.rfind("<script>")
    start = html.find("const DATA = ", script_at)
    if start == -1:
        raise SystemExit("Could not find the embedded DATA block in index.html")
    end = html.find("\n", start)
    html = html[:start] + line + html[end:]
    args.html.write_text(html, encoding="utf-8")

    print(
        f"Updated {args.html.name}: {len(payload['roles'])} roles, "
        f"{len(payload['tools'])} tools, {len(payload['district'])} districts, "
        f"{payload['metrics']['n_features']} features"
    )


if __name__ == "__main__":
    main()
