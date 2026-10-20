# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "numpy",
#     "pandas",
#     "pyarrow",
#     "statsmodels",
#     "scipy",
# ]
# ///
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARQUET = ROOT / "data" / "postings_v4_1019.parquet"
DEFAULT_OUT = ROOT / "analysis" / "out"

COLUMNS = [
    "uuid",
    "master_uuid",
    "metadata__newPostingDate",
    "month",
    "salary__minimum",
    "salary__maximum",
    "minimumYearsExperience",
    "role_entity_id",
    "role_label",
    "job_function",
    "tools",
    "positionLevels",
    "employmentTypes",
    "address__districts",
    "address__isOverseas",
    "is_recruiter",
    "is_new_role",
]

EXP_NAMES = [
    "exp",
    "exp_hinge_2",
    "exp_hinge_5",
    "exp_hinge_10",
    "exp_hinge_20",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--spec", choices=["role", "jf"], default="role")
    parser.add_argument("--top-tools", type=int, default=100)
    parser.add_argument("--min-salary", type=float, default=500)
    parser.add_argument("--max-salary", type=float, default=30000)
    parser.add_argument("--max-exp", type=int, default=40)
    parser.add_argument("--min-label-count", type=int, default=500)
    parser.add_argument("--min-district-count", type=int, default=200)
    return parser.parse_args()


def build_frame(path, min_salary, max_salary, max_exp):
    df = pd.read_parquet(path, columns=COLUMNS)
    counts = {"raw_rows": int(len(df))}
    df = df[df["uuid"] == df["master_uuid"]].copy()
    counts["master_rows"] = int(len(df))
    df["mid"] = (df["salary__minimum"] + df["salary__maximum"]) / 2.0
    df = df[df["mid"].between(min_salary, max_salary, inclusive="both")].copy()
    df = df[df["minimumYearsExperience"].between(0, max_exp, inclusive="both")].copy()
    counts["salary_exp_rows"] = int(len(df))
    df = df[df["role_entity_id"].notna()].copy()
    counts["role_rows"] = int(len(df))
    df = (
        df.sort_values("metadata__newPostingDate")
        .drop_duplicates("uuid", keep="last")
        .copy()
    )
    counts["model_rows"] = int(len(df))
    counts["unique_uuid"] = int(df["uuid"].nunique())
    df["log_mid"] = np.log(df["mid"])
    return df, counts


def parse_label_sets(series, field):
    return series.apply(
        lambda xs: frozenset(d[field] for d in xs) if xs is not None else frozenset()
    )


def drop_most_common(frame):
    if frame.shape[1] == 0:
        return frame, None
    base = frame.sum().idxmax()
    return frame.drop(columns=[base]), base


def one_hot(series, prefix, min_count=1):
    counts = series.value_counts()
    keep = counts[counts >= min_count].index
    values = series.where(series.isin(keep), "Other")
    frame = pd.get_dummies(values, prefix=prefix, dtype=np.int8)
    frame.columns = [f"{prefix}::{col.split('_', 1)[1]}" for col in frame.columns]
    return drop_most_common(frame)


def multilabel_flags(series, prefix, field, min_count):
    sets = parse_label_sets(series, field)
    counts = Counter()
    for labels in sets:
        counts.update(labels)
    keep = {label for label, n in counts.items() if n >= min_count}
    cols = {}
    for label in sorted(keep):
        cols[f"{prefix}::{label}"] = sets.apply(
            lambda s, value=label: int(value in s)
        ).to_numpy(dtype=np.int8)
    other = sets.apply(lambda s: int(any(label not in keep for label in s)))
    if other.sum():
        cols[f"{prefix}::Other"] = other.to_numpy(dtype=np.int8)
    missing = sets.apply(lambda s: int(len(s) == 0))
    if missing.sum():
        cols[f"{prefix}::Missing"] = missing.to_numpy(dtype=np.int8)
    frame = pd.DataFrame(cols, index=series.index)
    return drop_most_common(frame)


def district_series(series):
    def pick(districts):
        if districts is None or len(districts) == 0:
            return "Missing"
        locations = [d["location"] for d in districts]
        if any(location == "Islandwide" for location in locations):
            return "Islandwide"
        for location in locations:
            match = re.match(r"^(D\d{2})", location)
            if match:
                return match.group(1)
        return "Other"

    return series.apply(pick)


def build_tools(df, top_n):
    tool_sets = []
    for tools in df["tools"]:
        if tools is None:
            tool_sets.append(frozenset())
        else:
            tool_sets.append(frozenset(t["tool_label"] for t in tools))
    counts = Counter()
    for labels in tool_sets:
        counts.update(labels)
    top = [label for label, _ in counts.most_common(top_n)]
    positions = {label: i for i, label in enumerate(top)}
    matrix = np.zeros((len(df), len(top)), dtype=np.int8)
    for row, labels in enumerate(tool_sets):
        for label in labels:
            col = positions.get(label)
            if col is not None:
                matrix[row, col] = 1
    frame = pd.DataFrame(
        matrix, columns=[f"tool::{label}" for label in top], index=df.index
    )
    total_mentions = sum(counts.values())
    top_mentions = sum(counts[label] for label in top)
    rows_any = sum(any(label in positions for label in labels) for labels in tool_sets)
    stats = {
        "tool_mentions_total": int(total_mentions),
        "tools_flagged": len(top),
        "tool_mentions_flagged": int(top_mentions),
        "tool_mentions_flagged_pct": round(100 * top_mentions / total_mentions, 2)
        if total_mentions
        else 0.0,
        "rows_with_flagged_tool": int(rows_any),
        "rows_with_flagged_tool_pct": round(100 * rows_any / len(df), 2),
        "rows_with_any_tool": int(sum(1 for labels in tool_sets if labels)),
        "rows_with_any_tool_pct": round(
            100 * sum(1 for labels in tool_sets if labels) / len(df), 2
        ),
        "distinct_tools": int(len(counts)),
    }
    tool_counts = pd.Series({label: counts[label] for label in top})
    role_counts_by_tool = {}
    for labels, role in zip(tool_sets, df["role_label"].fillna("")):
        for label in labels:
            if label in positions:
                role_counts_by_tool.setdefault(label, Counter())[role] += 1
    diag_rows = []
    for label in top:
        role_counts = role_counts_by_tool.get(label, Counter())
        total = int(sum(role_counts.values()))
        top_role, top_n = role_counts.most_common(1)[0] if role_counts else ("", 0)
        diag_rows.append(
            {
                "tool": label,
                "top_role": top_role,
                "top_role_share": top_n / total if total else 0.0,
                "n_with_tool": total,
            }
        )
    tool_diag = pd.DataFrame(diag_rows)
    return frame, tool_counts, stats, tool_sets, tool_diag


def reduce_rank(design):
    values = design.to_numpy()
    _, triangular, pivot = qr(values, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(triangular))
    if len(diagonal) == 0:
        return design, []
    tolerance = diagonal[0] * max(values.shape) * np.finfo(float).eps
    rank = int((diagonal > tolerance).sum())
    if rank >= design.shape[1]:
        return design, []
    keep = sorted(pivot[:rank])
    dropped = [design.columns[i] for i in pivot[rank:]]
    return design.iloc[:, keep], dropped


def exp_basis(values):
    x = np.asarray(values, dtype=float)
    return np.column_stack(
        [
            x,
            np.maximum(x - 2, 0),
            np.maximum(x - 5, 0),
            np.maximum(x - 10, 0),
            np.maximum(x - 20, 0),
        ]
    )


def build_design(df, spec, args):
    tools_df, tool_counts, tool_stats, tool_sets, tool_diag = build_tools(
        df, args.top_tools
    )
    flagged_tools = set(tool_counts.index)
    tool_count = pd.Series(
        [len(labels - flagged_tools) for labels in tool_sets],
        index=df.index,
        name="rare_tool_count",
        dtype=np.int16,
    )
    if spec == "role":
        cat_df, cat_base = one_hot(df["role_entity_id"], "role")
        cat_counts = df["role_entity_id"].value_counts()
    else:
        cat_df, cat_base = one_hot(df["job_function"], "jf")
        cat_counts = df["job_function"].value_counts()
    pos_df, pos_base = multilabel_flags(
        df["positionLevels"], "pos", "position", args.min_label_count
    )
    emp_df, emp_base = multilabel_flags(
        df["employmentTypes"], "emp", "employmentType", args.min_label_count
    )
    dist_df, dist_base = one_hot(
        district_series(df["address__districts"]),
        "district",
        args.min_district_count,
    )
    month_df, month_base = one_hot(df["month"], "month")
    exp_df = pd.DataFrame(
        exp_basis(df["minimumYearsExperience"]), columns=EXP_NAMES, index=df.index
    )
    bool_df = pd.DataFrame(
        {
            "is_recruiter": df["is_recruiter"].astype(np.int8),
            "is_new_role": df["is_new_role"].astype(np.int8),
            "is_overseas": df["address__isOverseas"].astype(np.int8),
        },
        index=df.index,
    )
    design = pd.concat(
        [
            cat_df,
            exp_df,
            tools_df,
            tool_count,
            pos_df,
            emp_df,
            dist_df,
            month_df,
            bool_df,
        ],
        axis=1,
    )
    design = sm.add_constant(design, has_constant="add")
    nunique = design.nunique()
    zero_cols = nunique[(nunique <= 1) & (nunique.index != "const")].index
    if len(zero_cols):
        design = design.drop(columns=zero_cols)
    design = design.astype(np.float64)
    design, dropped_terms = reduce_rank(design)
    meta = {
        "category_prefix": spec,
        "category_base": cat_base,
        "category_counts": cat_counts,
        "position_base": pos_base,
        "employment_base": emp_base,
        "district_base": dist_base,
        "month_base": month_base,
        "dropped_terms": dropped_terms,
    }
    return design, tool_counts, tool_stats, meta, tool_diag


def coefficient_table(result):
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "term": result.params.index,
            "coef": result.params.to_numpy(),
            "std_err": result.bse.to_numpy(),
            "p_value": result.pvalues.to_numpy(),
            "ci_low_log": ci.iloc[:, 0].to_numpy(),
            "ci_high_log": ci.iloc[:, 1].to_numpy(),
        }
    )
    table["pct_effect"] = np.expm1(table["coef"])
    table["pct_ci_low"] = np.expm1(table["ci_low_log"])
    table["pct_ci_high"] = np.expm1(table["ci_high_log"])
    return table


def outcome_tables(table, spec, meta, tool_counts, tool_diag):
    if spec == "role":
        cat = table[table["term"].str.startswith("role::")].copy()
        cat["category_id"] = cat["term"].str.slice(len("role::"))
        cat["category_label"] = cat["category_id"].map(meta["role_labels"])
        cat["n_postings"] = cat["category_id"].map(meta["category_counts"])
        cat = cat.rename(
            columns={"category_id": "role_entity_id", "category_label": "role_label"}
        )
        cat_cols = [
            "role_entity_id",
            "role_label",
            "n_postings",
            "coef",
            "pct_effect",
            "pct_ci_low",
            "pct_ci_high",
            "std_err",
            "p_value",
        ]
    else:
        cat = table[table["term"].str.startswith("jf::")].copy()
        cat["job_function"] = cat["term"].str.slice(len("jf::"))
        cat["n_postings"] = cat["job_function"].map(meta["category_counts"])
        cat_cols = [
            "job_function",
            "n_postings",
            "coef",
            "pct_effect",
            "pct_ci_low",
            "pct_ci_high",
            "std_err",
            "p_value",
        ]
    cat = cat[cat_cols].sort_values("pct_effect", ascending=False)
    tools = table[table["term"].str.startswith("tool::")].copy()
    tools["tool"] = tools["term"].str.slice(len("tool::"))
    tools["n_postings"] = tools["tool"].map(tool_counts)
    tools = tools[
        [
            "tool",
            "n_postings",
            "coef",
            "pct_effect",
            "pct_ci_low",
            "pct_ci_high",
            "std_err",
            "p_value",
        ]
    ].sort_values("pct_effect", ascending=False)
    tools = tools.merge(tool_diag, on="tool", how="left")
    tools["nested_in_one_role"] = tools["top_role_share"] >= 0.95
    controls = table[
        ~table["term"].str.startswith(("role::", "jf::", "tool::"))
    ].copy()
    return cat, tools, controls


def experience_curve(result, design):
    beta = result.params.to_numpy()
    cov = result.cov_params().to_numpy()
    grid = np.arange(0, 21)
    basis = exp_basis(grid)
    rows = np.zeros((len(grid), design.shape[1]))
    rows[:, design.columns.get_loc("const")] = 1.0
    exp_cols = [design.columns.get_loc(name) for name in EXP_NAMES]
    rows[:, exp_cols] = basis
    log_pred = rows @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", rows, cov, rows))
    curve = pd.DataFrame(
        {
            "experience_years": grid,
            "predicted_mid": np.exp(log_pred),
            "ci_low": np.exp(log_pred - 1.96 * se),
            "ci_high": np.exp(log_pred + 1.96 * se),
        }
    )
    return curve


def format_pct(value):
    return f"{value * 100:+.1f}%"


def write_summary(path, spec, cat, tools, curve, metrics, meta):
    base = meta["category_base"]
    base_label = meta.get("category_base_label") or (
        base.split("::", 1)[1] if base else "baseline"
    )
    reference = ", ".join(
        [
            str(base_label),
            str(meta.get("position_base", "")).split("::", 1)[-1],
            str(meta.get("employment_base", "")).split("::", 1)[-1],
            str(meta.get("district_base", "")).split("::", 1)[-1],
            str(meta.get("month_base", "")).split("::", 1)[-1],
        ]
    )
    top_cat = cat.head(15).copy()
    bottom_cat = cat.tail(10).copy()
    top_tools = tools.head(15).copy()
    for frame in (top_cat, bottom_cat, top_tools):
        frame["pct_effect"] = frame["pct_effect"].map(format_pct)
        frame["pct_ci_low"] = frame["pct_ci_low"].map(format_pct)
        frame["pct_ci_high"] = frame["pct_ci_high"].map(format_pct)
    top_tools["top_role_share"] = top_tools["top_role_share"].map(
        lambda v: f"{v * 100:.0f}%"
    )
    curve_view = curve[curve["experience_years"].isin([0, 2, 5, 10, 15, 20])].copy()
    for col in ("predicted_mid", "ci_low", "ci_high"):
        curve_view[col] = curve_view[col].map(lambda v: f"{v:,.0f}")
    cat_name = "Role" if spec == "role" else "Job function"
    label_col = "role_label" if spec == "role" else "job_function"
    lines = [
        f"# Salary model: {cat_name.lower()} specification",
        "",
        "Target: `log((salary_min + salary_max) / 2)`.",
        f"Sample: {metrics['n_model_rows']:,} unique postings after master filter, salary/experience bounds, and uuid dedupe.",
        f"Coefficients are OLS with HC1 robust standard errors. No company clustering in this run.",
        "",
        "## Fit",
        "",
        f"- R-squared: {metrics['r2']:.4f}",
        f"- Adjusted R-squared: {metrics['adj_r2']:.4f}",
        f"- RMSE (log salary): {metrics['rmse_log']:.4f}",
        f"- MAE (log salary): {metrics['mae_log']:.4f}",
        f"- MAE (SGD, median prediction): {metrics['mae_sgd']:.0f}",
        f"- Parameters: {metrics['n_parameters']:,}",
        f"- Tools flagged: {metrics['tool_stats']['tools_flagged']} "
        f"({metrics['tool_stats']['tool_mentions_flagged_pct']}% of tool mentions)",
        "",
        f"## Highest {cat_name.lower()} effects",
        "",
        f"Percent differences vs `{base_label}` holding experience, tools, and controls fixed.",
        "",
        "```",
        top_cat[[label_col, "n_postings", "pct_effect", "pct_ci_low", "pct_ci_high", "p_value"]].to_string(index=False),
        "```",
        "",
        f"## Lowest {cat_name.lower()} effects",
        "",
        "```",
        bottom_cat[[label_col, "n_postings", "pct_effect", "pct_ci_low", "pct_ci_high", "p_value"]].to_string(index=False),
        "```",
        "",
        "## Tool effects",
        "",
        "Percent differences vs no flagged tool, holding role, experience, and controls fixed.",
        "Tools with a `top_role_share` near 100% are identified only from within-role variation; their coefficients are not cross-role premia.",
        "",
        "```",
        top_tools[["tool", "n_postings", "pct_effect", "pct_ci_low", "pct_ci_high", "p_value", "top_role", "top_role_share"]].to_string(index=False),
        "```",
        "",
        "## Experience curve",
        "",
        f"Predicted monthly midpoint at the reference profile ({reference}, no flagged tools).",
        "",
        "```",
        curve_view.to_string(index=False),
        "```",
        "",
        "## Caveats",
        "",
        "- These are conditional associations, not causal effects.",
        "- Only postings with advertised salary ranges are included.",
        "- Tool labels cover about half of postings; non-flagged tools enter only through `rare_tool_count`.",
        "- Tools concentrated in a single role (for example `delivery app`) are identified within that role only.",
        "- Role, tools, and experience are correlated; coefficients are conditional on the other terms.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df, counts = build_frame(
        args.parquet, args.min_salary, args.max_salary, args.max_exp
    )
    meta = {
        "role_labels": df.groupby("role_entity_id")["role_label"].agg(
            lambda s: s.dropna().iloc[0] if s.notna().any() else ""
        )
    }
    design, tool_counts, tool_stats, design_meta, tool_diag = build_design(
        df, args.spec, args
    )
    meta.update(design_meta)
    base_raw = meta["category_base"]
    base_id = base_raw.split("::", 1)[1] if base_raw else ""
    if args.spec == "role":
        meta["category_base_label"] = str(meta["role_labels"].get(base_id, base_id))
    else:
        meta["category_base_label"] = base_id
    y = df["log_mid"].to_numpy()
    result = sm.OLS(y, design).fit(cov_type="HC1")
    table = coefficient_table(result)
    cat, tools, controls = outcome_tables(
        table, args.spec, meta, tool_counts, tool_diag
    )
    curve = experience_curve(result, design)

    cat_file = "coef_roles.csv" if args.spec == "role" else "coef_jf.csv"
    cat.to_csv(args.out / cat_file, index=False, encoding="utf-8-sig")
    tools.to_csv(args.out / "coef_tools.csv", index=False, encoding="utf-8-sig")
    controls.to_csv(args.out / "coef_controls.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(args.out / "exp_curve.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "spec": args.spec,
        "target": "log((salary_min + salary_max) / 2)",
        "sample_counts": counts,
        "tool_stats": {
            **tool_stats,
            "tools_nested_in_one_role": int(tools["nested_in_one_role"].sum()),
        },
        "n_model_rows": int(len(df)),
        "n_parameters": int(design.shape[1]),
        "r2": float(result.rsquared),
        "adj_r2": float(result.rsquared_adj),
        "rmse_log": float(np.sqrt(np.mean(result.resid**2))),
        "mae_log": float(np.mean(np.abs(result.resid))),
        "mae_sgd": float(
            np.mean(np.abs(np.exp(result.fittedvalues) - df["mid"].to_numpy()))
        ),
        "n_roles": int(df["role_entity_id"].nunique()),
        "n_job_functions": int(df["job_function"].nunique()),
        "salary_mid_median": float(df["mid"].median()),
        "salary_mid_p25": float(df["mid"].quantile(0.25)),
        "salary_mid_p75": float(df["mid"].quantile(0.75)),
        "experience_median": float(df["minimumYearsExperience"].median()),
        "baseline": {
            "category": meta["category_base"],
            "category_label": meta["category_base_label"],
            "position": meta["position_base"],
            "employment": meta["employment_base"],
            "district": meta["district_base"],
            "month": meta["month_base"],
        },
    }
    (args.out / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_summary(
        args.out / "summary.md", args.spec, cat, tools, curve, metrics, meta
    )

    label_col = "role_label" if args.spec == "role" else "job_function"
    print(
        f"rows={len(df):,} params={design.shape[1]:,} "
        f"R2={result.rsquared:.4f} adjR2={result.rsquared_adj:.4f}"
    )
    print(f"tools flagged={tool_stats['tools_flagged']} "
          f"mentions covered={tool_stats['tool_mentions_flagged_pct']}%")
    print("\nTop effects:")
    print(
        cat.head(12)[[label_col, "n_postings", "pct_effect", "p_value"]]
        .assign(pct_effect=lambda d: d["pct_effect"].map(format_pct))
        .to_string(index=False)
    )
    print("\nBottom effects:")
    print(
        cat.tail(8)[[label_col, "n_postings", "pct_effect", "p_value"]]
        .assign(pct_effect=lambda d: d["pct_effect"].map(format_pct))
        .to_string(index=False)
    )
    print("\nTop tools:")
    print(
        tools.head(12)[["tool", "n_postings", "pct_effect", "p_value"]]
        .assign(pct_effect=lambda d: d["pct_effect"].map(format_pct))
        .to_string(index=False)
    )
    print("\nExperience curve:")
    print(
        curve[curve["experience_years"].isin([0, 2, 5, 10, 15, 20])]
        .assign(
            predicted_mid=lambda d: d["predicted_mid"].map(lambda v: f"{v:,.0f}")
        )
        .to_string(index=False)
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
