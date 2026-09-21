"""
sanity_checks.py
----------------
Pre-analysis sanity checks every experiment must pass before results are trusted.

1. Sample Ratio Mismatch (SRM) — was the traffic split actually 50/50?
2. A/A Test — do pre-experiment metrics look the same across variants?
3. Novelty effect detection — does the treatment effect decay over time?

If SRM is detected, the experiment result is INVALID regardless of p-value.
This is one of the most common production bugs in experimentation platforms.

Usage:
    from src.sanity_checks import check_srm, check_aa, novelty_check
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict


# ── 1. Sample Ratio Mismatch (SRM) ───────────────────────────────────────────

def check_srm(
    n_control: int,
    n_treatment: int,
    expected_split: float = 0.50,
    alpha: float = 0.01,   # Use 0.01 for SRM — we want to be conservative
) -> Dict:
    """
    Chi-square test for whether observed traffic split matches expected split.

    SRM happens when randomization breaks — e.g. a bug sends 48% to treatment
    instead of 50%. Even small SRM invalidates the experiment because the groups
    may differ systematically (e.g. only certain device types got into treatment).

    Returns a dict with test result and recommendation.
    """
    n_total   = n_control + n_treatment
    expected_ctrl  = n_total * (1 - expected_split)
    expected_treat = n_total * expected_split

    chi2, p_value = stats.chisquare(
        f_obs=[n_control, n_treatment],
        f_exp=[expected_ctrl, expected_treat],
    )

    actual_split  = n_treatment / n_total
    split_delta   = abs(actual_split - expected_split)

    srm_detected  = p_value < alpha

    return {
        "n_control":       n_control,
        "n_treatment":     n_treatment,
        "n_total":         n_total,
        "expected_split":  expected_split,
        "actual_split":    round(actual_split, 5),
        "split_delta_pp":  round(split_delta * 100, 3),
        "chi2_stat":       round(chi2, 4),
        "p_value":         round(p_value, 6),
        "alpha":           alpha,
        "srm_detected":    srm_detected,
        "verdict":         "⚠️ SRM DETECTED — experiment results are INVALID" if srm_detected
                           else "✅ No SRM — traffic split looks correct",
        "recommendation":  "Halt analysis and investigate randomization bug" if srm_detected
                           else "Proceed with analysis",
    }


def srm_summary_df(srm_result: Dict) -> pd.DataFrame:
    """Format SRM result as a clean DataFrame for notebook display."""
    rows = [
        ("Control users",       f"{srm_result['n_control']:,}"),
        ("Treatment users",     f"{srm_result['n_treatment']:,}"),
        ("Expected split",      f"{srm_result['expected_split']:.0%}"),
        ("Actual split (treat)",f"{srm_result['actual_split']:.3%}"),
        ("Split delta",         f"{srm_result['split_delta_pp']:.3f}pp"),
        ("Chi² statistic",      f"{srm_result['chi2_stat']:.4f}"),
        ("p-value",             f"{srm_result['p_value']:.6f}"),
        ("Verdict",             srm_result["verdict"]),
    ]
    return pd.DataFrame(rows, columns=["Check", "Result"])


# ── 2. A/A Test (Pre-experiment Metric Parity) ────────────────────────────────

def check_aa(
    control_values: pd.Series,
    treatment_values: pd.Series,
    metric_name: str,
    alpha: float = 0.05,
) -> Dict:
    """
    Verify that control and treatment groups had similar pre-experiment metrics.
    This uses Day 0 data BEFORE the treatment was applied.

    If pre-experiment metrics differ significantly, it suggests the randomization
    didn't properly balance the groups — another sign of an experiment bug.
    """
    ctrl_mean  = control_values.mean()
    treat_mean = treatment_values.mean()
    diff       = treat_mean - ctrl_mean
    pct_diff   = diff / ctrl_mean if ctrl_mean != 0 else 0

    t_stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)

    issue_detected = p_value < alpha

    return {
        "metric":          metric_name,
        "control_mean":    round(ctrl_mean, 5),
        "treatment_mean":  round(treat_mean, 5),
        "difference":      round(diff, 5),
        "pct_difference":  round(pct_diff * 100, 3),
        "t_stat":          round(t_stat, 4),
        "p_value":         round(p_value, 6),
        "issue_detected":  issue_detected,
        "verdict":         f"⚠️ Pre-experiment imbalance on {metric_name}" if issue_detected
                           else f"✅ {metric_name} balanced pre-experiment",
    }


def aa_summary_df(aa_results: list) -> pd.DataFrame:
    """Format multiple A/A check results as a summary table."""
    rows = []
    for r in aa_results:
        rows.append({
            "metric":          r["metric"],
            "control_mean":    r["control_mean"],
            "treatment_mean":  r["treatment_mean"],
            "pct_diff":        f"{r['pct_difference']:+.3f}%",
            "p_value":         r["p_value"],
            "verdict":         "⚠️ Imbalanced" if r["issue_detected"] else "✅ Balanced",
        })
    return pd.DataFrame(rows)


# ── 3. Novelty Effect Check ───────────────────────────────────────────────────

def novelty_check(
    daily_df: pd.DataFrame,
    metric: str,
    early_days: int = 5,
    late_days_start: int = 7,
) -> Dict:
    """
    Compare treatment lift in early days vs. later days.
    If the lift is significantly larger early on and decays, novelty effect is present.

    daily_df must have columns: day, variant, and the metric column.
    """
    ctrl  = daily_df[daily_df["variant"] == "control"]
    treat = daily_df[daily_df["variant"] == "treatment"]

    def daily_mean(df):
        return df.groupby("day")[metric].mean()

    ctrl_daily  = daily_mean(ctrl)
    treat_daily = daily_mean(treat)
    lift_daily  = (treat_daily - ctrl_daily) / ctrl_daily

    early_lift = lift_daily[lift_daily.index < early_days].mean()
    late_lift  = lift_daily[lift_daily.index >= late_days_start].mean()
    decay      = early_lift - late_lift

    novelty_suspected = (early_lift > 0) and (decay > early_lift * 0.30)

    return {
        "metric":             metric,
        "early_lift_pct":     round(early_lift * 100, 3),
        "late_lift_pct":      round(late_lift * 100, 3),
        "decay_pp":           round(decay * 100, 3),
        "novelty_suspected":  novelty_suspected,
        "verdict": (
            f"⚠️ Novelty effect suspected — early lift ({early_lift*100:.1f}%) "
            f"decays to ({late_lift*100:.1f}%) after day {late_days_start}"
            if novelty_suspected
            else f"✅ No novelty effect — lift is stable across experiment duration"
        ),
        "lift_by_day": lift_daily,
    }
