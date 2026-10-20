# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "statsmodels",
#     "scipy",
#     "scikit-learn",
# ]
# ///
import argparse
import json
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from salary_model import (
    DEFAULT_PARQUET,
    DEFAULT_OUT,
    EXP_NAMES,
    build_frame,
    district_series,
    exp_basis,
    multilabel_flags,
    one_hot,
    parse_label_sets,
    reduce_rank,
    format_pct,
)

def count_tools(tool_sets):
    counts = Counter()
    for labels in tool_sets:
        counts.update(labels)
    return counts


def build_design(df, args, tool_list):
    # role dummies
    role_counts = df["role_entity_id"].value_counts()
    keep_roles = role_counts[role_counts >= args.min_role_n].index
    role_vals = df["role_entity_id"].where(df["role_entity_id"].isin(keep_roles), "Other")
    role_df = pd.get_dummies(role_vals, prefix="role", dtype=np.int8)
    role_df.columns = [f"role::{c.split('_', 1)[1]}" for c in role_df.columns]
    base_role = role_df.sum().idxmax()
    role_df = role_df.drop(columns=[base_role])
    role_labels = df.groupby("role_entity_id")["role_label"].agg(
        lambda s: s.dropna().iloc[0] if s.notna().any() else ""
    )

    exp_df = pd.DataFrame(exp_basis(df["minimumYearsExperience"]), columns=EXP_NAMES, index=df.index)

    pos_df, _ = multilabel_flags(df["positionLevels"], "pos", "position", args.min_label_count)
    emp_df, _ = multilabel_flags(df["employmentTypes"], "emp", "employmentType", args.min_label_count)
    dist_df, _ = one_hot(district_series(df["address__districts"]), "district", args.min_district_count)

    # FDR tool flags
    tool_sets = parse_label_sets(df["tools"], "tool_label")
    positions = {t: i for i, t in enumerate(sorted(tool_list))}
    matrix = np.zeros((len(df), len(tool_list)), dtype=np.int8)
    for row, labels in enumerate(tool_sets):
        for t in labels:
            col = positions.get(t)
            if col is not None:
                matrix[row, col] = 1
    tools_df = pd.DataFrame(matrix, columns=[f"tool::{t}" for t in sorted(tool_list)], index=df.index)

    parts = [role_df, exp_df]
    if pos_df.shape[1]:
        parts.append(pos_df)
    if emp_df.shape[1]:
        parts.append(emp_df)
    if dist_df.shape[1]:
        parts.append(dist_df)
    parts.append(tools_df)
    design = pd.concat(parts, axis=1)
    design = sm.add_constant(design, has_constant="add")
    nunique = design.nunique()
    drop_cols = nunique[(nunique <= 1) & (nunique.index != "const")].index
    if len(drop_cols):
        design = design.drop(columns=drop_cols)
    design = design.astype(np.float64)
    design, dropped = reduce_rank(design)
    meta = {
        "base_role": base_role,
        "role_labels": role_labels,
        "position_base": None,
        "employment_base": None,
        "district_base": None,
        "dropped": dropped,
    }
    # capture bases
    pos_cols = [c for c in design.columns if c.startswith("pos::")]
    emp_cols = [c for c in design.columns if c.startswith("emp::")]
    dist_cols = [c for c in design.columns if c.startswith("district::")]
    meta["position_base"] = None
    meta["employment_base"] = None
    meta["district_base"] = None
    return design, meta, tool_sets


def select_fdr_tools(df, args):
    """Screen the most common tools with Benjamini-Hochberg FDR.

    All candidates enter one OLS together with the structural controls; tools
    whose HC1 p-value survives the FDR threshold are kept for the final model.
    """
    tool_sets = parse_label_sets(df["tools"], "tool_label")
    counts = count_tools(tool_sets)
    candidates = [t for t, _ in counts.most_common(args.tool_candidates)]
    if not candidates:
        return [], candidates, pd.Series(dtype=float)

    design, _, _ = build_design(df, args, candidates)
    y = df["log_mid"].to_numpy()
    fit = sm.OLS(y, design).fit(cov_type="HC1")
    terms = [f"tool::{t}" for t in candidates if f"tool::{t}" in design.columns]
    pvals = fit.pvalues[terms].to_numpy()
    reject, qvals, _, _ = multipletests(pvals, alpha=args.fdr_q, method="fdr_bh")
    selected = sorted({t.replace("tool::", "") for t, keep in zip(terms, reject) if keep})
    report = pd.DataFrame(
        {
            "tool": [t.replace("tool::", "") for t in terms],
            "n_postings": [counts.get(t.replace("tool::", ""), 0) for t in terms],
            "coef": fit.params[terms].to_numpy(),
            "p_value": pvals,
            "q_value": qvals,
            "selected": reject,
        }
    ).sort_values("q_value")
    print(
        f"FDR screen: {len(selected)} of {len(candidates)} candidates kept "
        f"(q<={args.fdr_q}); {len(candidates) - len(selected)} dropped"
    )
    return selected, candidates, report


def group_importance(df, design, train_mask, test_mask, seed=42, repeats=5):
    """Permutation importance by feature group: holdout MAE increase in SGD."""
    xtr = design.loc[train_mask].to_numpy(dtype=float)
    xte = design.loc[test_mask].to_numpy(dtype=float)
    ytr = df.loc[train_mask, "log_mid"].to_numpy()
    yte = df.loc[test_mask, "log_mid"].to_numpy()
    coef, *_ = np.linalg.lstsq(xtr, ytr, rcond=None)
    baseline = float(np.mean(np.abs(np.exp(np.clip(xte @ coef, -10, 10)) - np.exp(yte))))

    groups = {
        "Role": [i for i, c in enumerate(design.columns) if c.startswith("role::")],
        "Experience": [i for i, c in enumerate(design.columns) if c in EXP_NAMES],
        "Position level": [i for i, c in enumerate(design.columns) if c.startswith("pos::")],
        "Employment type": [i for i, c in enumerate(design.columns) if c.startswith("emp::")],
        "Tools": [i for i, c in enumerate(design.columns) if c.startswith("tool::")],
        "District": [i for i, c in enumerate(design.columns) if c.startswith("district::")],
    }
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in groups.items():
        if not cols:
            continue
        deltas = []
        for _ in range(repeats):
            shuffled = xte.copy()
            shuffled[:, cols] = xte[rng.permutation(len(xte))][:, cols]
            mae = float(np.mean(np.abs(np.exp(np.clip(shuffled @ coef, -10, 10)) - np.exp(yte))))
            deltas.append(mae - baseline)
        rows.append({"group": name, "delta_mae_sgd": float(np.mean(deltas))})
    return rows


def cv_metrics(df, design, n_splits=5):
    y = df["log_mid"].to_numpy()
    uen = df["postedCompany__uen"].to_numpy()
    x = design.to_numpy(dtype=float)
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=n_splits)
    r2s, mae_logs, mae_sgds = [], [], []
    for train_idx, test_idx in gkf.split(x, y, groups=uen):
        coef, *_ = np.linalg.lstsq(x[train_idx], y[train_idx], rcond=None)
        pred = np.clip(x[test_idx] @ coef, -10, 10)
        resid = pred - y[test_idx]
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y[test_idx] - y[test_idx].mean()) ** 2))
        r2s.append(1 - ss_res / ss_tot)
        mae_logs.append(float(np.mean(np.abs(resid))))
        mae_sgds.append(float(np.mean(np.abs(np.exp(pred) - np.exp(y[test_idx])))))
    return {"n_features": design.shape[1],
            "cv_r2": float(np.mean(r2s)), "cv_r2_std": float(np.std(r2s)),
            "cv_mae_log": float(np.mean(mae_logs)), "cv_mae_sgd": float(np.mean(mae_sgds))}


def holdout_eval(df, design, train_mask, test_mask):
    xtr = design.loc[train_mask].to_numpy(dtype=float)
    xte = design.loc[test_mask].to_numpy(dtype=float)
    ytr = df.loc[train_mask, "log_mid"].to_numpy()
    yte = df.loc[test_mask, "log_mid"].to_numpy()
    coef, *_ = np.linalg.lstsq(xtr, ytr, rcond=None)
    pred = np.clip(xte @ coef, -10, 10)
    r2 = 1 - np.sum((pred - yte) ** 2) / np.sum((yte - yte.mean()) ** 2)
    mae_log = float(np.mean(np.abs(pred - yte)))
    mae_sgd = float(np.mean(np.abs(np.exp(pred) - np.exp(yte))))
    return float(r2), mae_log, mae_sgd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-role-n", type=int, default=400)
    parser.add_argument("--min-label-count", type=int, default=500)
    parser.add_argument("--min-district-count", type=int, default=200)
    parser.add_argument("--max-exp", type=int, default=40)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--tool-candidates", type=int, default=100)
    parser.add_argument("--fdr-q", type=float, default=0.05)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, counts = build_frame(args.parquet, 500, 30000, args.max_exp)
    df = df.reset_index(drop=True)
    meta_uen = pd.read_parquet(args.parquet, columns=["uuid", "postedCompany__uen"])
    meta_uen = meta_uen.drop_duplicates("uuid").set_index("uuid")
    df["postedCompany__uen"] = df["uuid"].map(meta_uen["postedCompany__uen"]).fillna("unknown")

    selected_tools, candidates, fdr_report = select_fdr_tools(df, args)
    fdr_report.to_csv(args.out / "final_model_tool_screen.csv", index=False, encoding="utf-8-sig")
    design, meta, tool_sets = build_design(df, args, selected_tools)

    # holdout split
    rng = np.random.default_rng(42)
    uens = df["postedCompany__uen"].unique()
    rng.shuffle(uens)
    n_hold = int(len(uens) * args.holdout_frac)
    hold_uens = set(uens[:n_hold])
    train_mask = ~df["postedCompany__uen"].isin(hold_uens)
    test_mask = df["postedCompany__uen"].isin(hold_uens)

    cv = cv_metrics(df, design, args.n_splits)
    r2, ml, ms = holdout_eval(df, design, train_mask, test_mask)

    # OLS with HC1 for coefficients (in-sample, full data)
    y = df["log_mid"].to_numpy()
    ols = sm.OLS(y, design).fit(cov_type="HC1")
    ci = ols.conf_int()
    coef_table = pd.DataFrame({
        "term": ols.params.index,
        "coef": ols.params.to_numpy(),
        "std_err": ols.bse.to_numpy(),
        "p_value": ols.pvalues.to_numpy(),
        "ci_low_log": ci.iloc[:, 0].to_numpy(),
        "ci_high_log": ci.iloc[:, 1].to_numpy(),
    })
    coef_table["pct_effect"] = np.expm1(coef_table["coef"])
    coef_table["pct_ci_low"] = np.expm1(coef_table["ci_low_log"])
    coef_table["pct_ci_high"] = np.expm1(coef_table["ci_high_log"])
    coef_table.to_csv(args.out / "final_model_coefficients.csv", index=False, encoding="utf-8-sig")

    # role coefficients
    role_coefs = coef_table[coef_table["term"].str.startswith("role::")].copy()
    role_coefs["role_entity_id"] = role_coefs["term"].str.slice(len("role::"))
    role_coefs["role_label"] = role_coefs["role_entity_id"].map(meta["role_labels"])
    role_coefs["n_postings"] = role_coefs["role_entity_id"].map(df["role_entity_id"].value_counts())
    role_coefs = role_coefs[["role_entity_id", "role_label", "n_postings", "coef", "pct_effect",
                             "pct_ci_low", "pct_ci_high", "std_err", "p_value"]]
    role_coefs = role_coefs.sort_values("pct_effect", ascending=False)
    role_coefs.to_csv(args.out / "final_model_roles.csv", index=False, encoding="utf-8-sig")

    # tool coefficients
    tool_coefs = coef_table[coef_table["term"].str.startswith("tool::")].copy()
    tool_coefs["tool"] = tool_coefs["term"].str.slice(len("tool::"))
    tool_coefs["n_postings"] = tool_coefs["tool"].map(
        pd.Series({t: sum(1 for ts in tool_sets if t in ts) for t in selected_tools})
    )
    tool_coefs = tool_coefs[["tool", "n_postings", "coef", "pct_effect", "pct_ci_low",
                             "pct_ci_high", "std_err", "p_value"]]
    tool_coefs = tool_coefs.sort_values("pct_effect", ascending=False)
    tool_coefs.to_csv(args.out / "final_model_tools.csv", index=False, encoding="utf-8-sig")

    # control coefficients
    control_coefs = coef_table[
        ~coef_table["term"].str.startswith(("role::", "tool::"))
    ].copy()
    control_coefs.to_csv(args.out / "final_model_controls.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "model": (
            f"FDR tools ({len(selected_tools)} of {len(candidates)}) + role + "
            "experience + position + employment + district"
        ),
        "target": "log((salary_min + salary_max) / 2)",
        "n_features": int(design.shape[1]),
        "n_tool_candidates": int(len(candidates)),
        "n_tools_selected": int(len(selected_tools)),
        "fdr_q": float(args.fdr_q),
        "dropped_collinear": meta["dropped"],
        "sample_counts": counts,
        "n_model_rows": int(len(df)),
        "n_roles": int(df["role_entity_id"].nunique()),
        "salary_mid_median": float(df["mid"].median()) if "mid" in df.columns else None,
        "cv_r2": cv["cv_r2"],
        "cv_r2_std": cv["cv_r2_std"],
        "cv_mae_log": cv["cv_mae_log"],
        "cv_mae_sgd": cv["cv_mae_sgd"],
        "holdout_r2": r2,
        "holdout_mae_log": ml,
        "holdout_mae_sgd": ms,
        "in_sample_r2": float(ols.rsquared),
        "in_sample_adj_r2": float(ols.rsquared_adj),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "features": {
            "role": True, "experience_spline": True, "position": True,
            "employment": True, "district": True, "tools_fdr_selected": True,
            "month": False, "bools": False, "rare_tool_count": False,
        },
        "group_importance": group_importance(df, design, train_mask, test_mask),
    }
    # fix mid
    df["mid"] = (df["salary__minimum"] + df["salary__maximum"]) / 2.0
    metrics["salary_mid_median"] = float(df["mid"].median())
    metrics["salary_mid_p25"] = float(df["mid"].quantile(0.25))
    metrics["salary_mid_p75"] = float(df["mid"].quantile(0.75))
    (args.out / "final_model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # experience curve
    beta = ols.params.to_numpy()
    cov = ols.cov_params().to_numpy()
    grid = np.arange(0, 21)
    basis = exp_basis(grid)
    rows = np.zeros((len(grid), design.shape[1]))
    rows[:, design.columns.get_loc("const")] = 1.0
    exp_cols = [design.columns.get_loc(name) for name in EXP_NAMES]
    rows[:, exp_cols] = basis
    log_pred = rows @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", rows, cov, rows))
    curve = pd.DataFrame({
        "experience_years": grid,
        "predicted_mid": np.exp(log_pred),
        "ci_low": np.exp(log_pred - 1.96 * se),
        "ci_high": np.exp(log_pred + 1.96 * se),
    })
    curve.to_csv(args.out / "final_model_exp_curve.csv", index=False, encoding="utf-8-sig")

    print(f"FINAL MODEL: {design.shape[1]} features")
    print(f"  CV:   R2={cv['cv_r2']:.4f} (sd {cv['cv_r2_std']:.4f})  MAE={cv['cv_mae_sgd']:.1f} SGD")
    print(f"  Hold: R2={r2:.4f}  MAE={ms:.1f} SGD  (n_test={int(test_mask.sum())})")
    print(f"  In-sample: R2={ols.rsquared:.4f} adj={ols.rsquared_adj:.4f}")
    _base = meta["base_role"]
    _base_id = _base.split("::", 1)[1] if "::" in _base else _base
    _base_label = meta["role_labels"].get(_base_id, _base_id)
    print(f"\nTop 10 role effects (vs {_base_label}):")
    print(role_coefs.head(10)[["role_label", "n_postings", "pct_effect", "p_value"]]
          .assign(pct_effect=lambda d: d["pct_effect"].map(format_pct)).to_string(index=False))
    print(f"\nTop 10 tool effects:")
    print(tool_coefs.head(10)[["tool", "n_postings", "pct_effect", "p_value"]]
          .assign(pct_effect=lambda d: d["pct_effect"].map(format_pct)).to_string(index=False))
    print(f"\nExperience: 0y={curve.iloc[0]['predicted_mid']:,.0f}  5y={curve.iloc[5]['predicted_mid']:,.0f}  10y={curve.iloc[10]['predicted_mid']:,.0f}  20y={curve.iloc[20]['predicted_mid']:,.0f}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
