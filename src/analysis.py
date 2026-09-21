"""
analysis.py
-----------
Core statistical analysis for the A/B test:
  - Two-sample t-test for continuous metrics
  - Z-test for proportions (retention)
  - Delta method for ratio metrics (engagement rate = comments / sessions)
  - Confidence interval construction
  - Multiple testing correction (Bonferroni)
  - Segment breakdown analysis
  - Ship / No-Ship decision framework

Usage:
    from src.analysis import test_metric, run_full_analysis, ship_decision
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple


# ── Two-Sample T-Test ─────────────────────────────────────────────────────────

def test_metric(
    control: pd.Series,
    treatment: pd.Series,
    metric_name: str,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> Dict:
    """
    Two-sample Welch's t-test (unequal variance — safer default).
    Returns full result dict with effect size, CI, and verdict.
    """
    ctrl_mean  = control.mean()
    treat_mean = treatment.mean()
    ctrl_std   = control.std()
    treat_std  = treatment.std()
    n_ctrl     = len(control)
    n_treat    = len(treatment)

    abs_diff   = treat_mean - ctrl_mean
    rel_diff   = abs_diff / ctrl_mean if ctrl_mean != 0 else 0

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
    if not two_sided:
        p_value = p_value / 2

    # 95% confidence interval on absolute difference
    se_diff = np.sqrt(ctrl_std**2 / n_ctrl + treat_std**2 / n_treat)
    z_crit  = stats.norm.ppf(1 - alpha / 2)
    ci_low  = abs_diff - z_crit * se_diff
    ci_high = abs_diff + z_crit * se_diff

    # Cohen's d effect size
    pooled_std = np.sqrt((ctrl_std**2 + treat_std**2) / 2)
    cohens_d   = abs_diff / pooled_std if pooled_std > 0 else 0

    significant = p_value < alpha

    return {
        "metric":        metric_name,
        "ctrl_mean":     round(ctrl_mean, 5),
        "treat_mean":    round(treat_mean, 5),
        "abs_diff":      round(abs_diff, 5),
        "rel_diff_pct":  round(rel_diff * 100, 3),
        "ci_low":        round(ci_low, 5),
        "ci_high":       round(ci_high, 5),
        "t_stat":        round(t_stat, 4),
        "p_value":       round(p_value, 6),
        "cohens_d":      round(cohens_d, 4),
        "alpha":         alpha,
        "significant":   significant,
        "direction":     "positive" if abs_diff > 0 else "negative",
        "n_ctrl":        n_ctrl,
        "n_treat":       n_treat,
    }


# ── Z-Test for Proportions ────────────────────────────────────────────────────

def test_proportion(
    n_ctrl: int,
    n_treat: int,
    success_ctrl: int,
    success_treat: int,
    metric_name: str,
    alpha: float = 0.05,
) -> Dict:
    """
    Two-proportion z-test for binary metrics (e.g. D7 retention).
    """
    p_ctrl  = success_ctrl  / n_ctrl
    p_treat = success_treat / n_treat
    diff    = p_treat - p_ctrl
    rel_diff = diff / p_ctrl if p_ctrl > 0 else 0

    p_pool = (success_ctrl + success_treat) / (n_ctrl + n_treat)
    se     = np.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_treat))
    z_stat = diff / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    z_crit  = stats.norm.ppf(1 - alpha / 2)
    se_diff = np.sqrt(p_ctrl*(1-p_ctrl)/n_ctrl + p_treat*(1-p_treat)/n_treat)
    ci_low  = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    return {
        "metric":        metric_name,
        "ctrl_mean":     round(p_ctrl, 5),
        "treat_mean":    round(p_treat, 5),
        "abs_diff":      round(diff, 5),
        "rel_diff_pct":  round(rel_diff * 100, 3),
        "ci_low":        round(ci_low, 5),
        "ci_high":       round(ci_high, 5),
        "z_stat":        round(z_stat, 4),
        "p_value":       round(p_value, 6),
        "alpha":         alpha,
        "significant":   p_value < alpha,
        "direction":     "positive" if diff > 0 else "negative",
        "n_ctrl":        n_ctrl,
        "n_treat":       n_treat,
    }


# ── Delta Method for Ratio Metrics ────────────────────────────────────────────

def test_ratio_metric(
    df: pd.DataFrame,
    numerator_col: str,
    denominator_col: str,
    metric_name: str,
    alpha: float = 0.05,
) -> Dict:
    """
    Delta method for ratio metrics like engagement rate (comments / sessions).

    Standard t-test is WRONG for ratio metrics because it ignores the correlation
    between numerator and denominator. The delta method approximates the variance
    of a ratio properly.

    This is a common interview question at Meta — knowing this separates
    candidates who've actually run experiments from those who just read about them.
    """
    ctrl  = df[df["variant"] == "control"]
    treat = df[df["variant"] == "treatment"]

    def ratio_stats(grp):
        num  = grp[numerator_col]
        den  = grp[denominator_col]
        n    = len(grp)
        r    = num.mean() / den.mean()
        # Delta method variance: Var(N/D) ≈ (1/D̄)²·Var(N) + (N̄/D̄²)²·Var(D) − 2·(N̄/D̄³)·Cov(N,D)
        var_num = num.var() / n
        var_den = den.var() / n
        cov_nd  = np.cov(num, den)[0, 1] / n
        var_r   = (1/den.mean())**2 * var_num + \
                  (num.mean()/den.mean()**2)**2 * var_den - \
                  2 * (num.mean()/den.mean()**3) * cov_nd
        return r, np.sqrt(var_r), n

    r_ctrl,  se_ctrl,  n_ctrl  = ratio_stats(ctrl)
    r_treat, se_treat, n_treat = ratio_stats(treat)

    diff     = r_treat - r_ctrl
    rel_diff = diff / r_ctrl if r_ctrl > 0 else 0
    se_diff  = np.sqrt(se_ctrl**2 + se_treat**2)
    z_stat   = diff / se_diff if se_diff > 0 else 0
    p_value  = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    z_crit  = stats.norm.ppf(1 - alpha / 2)
    ci_low  = diff - z_crit * se_diff
    ci_high = diff + z_crit * se_diff

    return {
        "metric":        metric_name,
        "ctrl_mean":     round(r_ctrl, 5),
        "treat_mean":    round(r_treat, 5),
        "abs_diff":      round(diff, 5),
        "rel_diff_pct":  round(rel_diff * 100, 3),
        "ci_low":        round(ci_low, 5),
        "ci_high":       round(ci_high, 5),
        "z_stat":        round(z_stat, 4),
        "p_value":       round(p_value, 6),
        "alpha":         alpha,
        "significant":   p_value < alpha,
        "direction":     "positive" if diff > 0 else "negative",
        "n_ctrl":        n_ctrl,
        "n_treat":       n_treat,
        "method":        "delta method",
    }


# ── Multiple Testing Correction ───────────────────────────────────────────────

def bonferroni_correction(results: List[Dict], family_alpha: float = 0.05) -> List[Dict]:
    """
    Apply Bonferroni correction for multiple comparisons.
    Adjusted alpha = family_alpha / number of tests.
    Each result dict gets an 'adjusted_significant' field added.
    """
    n_tests       = len(results)
    adjusted_alpha = family_alpha / n_tests

    for r in results:
        r["bonferroni_alpha"]        = round(adjusted_alpha, 5)
        r["adjusted_significant"]    = r["p_value"] < adjusted_alpha

    return results


# ── Full Analysis Summary ─────────────────────────────────────────────────────

def results_summary_df(results: List[Dict]) -> pd.DataFrame:
    """
    Build a clean summary table from a list of metric test result dicts.
    """
    rows = []
    for r in results:
        sig = r.get("adjusted_significant", r.get("significant", False))
        rows.append({
            "metric":          r["metric"],
            "control_mean":    r["ctrl_mean"],
            "treatment_mean":  r["treat_mean"],
            "lift_pct":        f"{r['rel_diff_pct']:+.2f}%",
            "95% CI":          f"[{r['rel_diff_pct'] - abs(r['ci_high'] - r['abs_diff']) / r['ctrl_mean'] * 100:.2f}%, "
                               f"{r['rel_diff_pct'] + abs(r['ci_high'] - r['abs_diff']) / r['ctrl_mean'] * 100:.2f}%]"
                               if r['ctrl_mean'] != 0 else "N/A",
            "p_value":         r["p_value"],
            "significant":     "✅ Yes" if sig else "❌ No",
            "direction":       r["direction"],
        })
    return pd.DataFrame(rows)


# ── Segment Analysis ──────────────────────────────────────────────────────────

def segment_analysis(
    df: pd.DataFrame,
    metric: str,
    segment_col: str = "segment",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Break down treatment effect by user segment.
    Useful for finding heterogeneous treatment effects —
    e.g. the feature helps power users but hurts new users.
    """
    rows = []
    for seg in df[segment_col].unique():
        seg_df = df[df[segment_col] == seg]
        ctrl   = seg_df[seg_df["variant"] == "control"][metric]
        treat  = seg_df[seg_df["variant"] == "treatment"][metric]
        result = test_metric(ctrl, treat, metric_name=f"{metric} — {seg}", alpha=alpha)
        rows.append({
            "segment":      seg,
            "n_ctrl":       result["n_ctrl"],
            "n_treat":      result["n_treat"],
            "ctrl_mean":    result["ctrl_mean"],
            "treat_mean":   result["treat_mean"],
            "lift_pct":     f"{result['rel_diff_pct']:+.2f}%",
            "p_value":      result["p_value"],
            "significant":  "✅" if result["significant"] else "❌",
        })
    return pd.DataFrame(rows).sort_values("segment")


# ── Ship / No-Ship Decision Framework ─────────────────────────────────────────

def ship_decision(
    primary_results: List[Dict],
    guardrail_results: List[Dict],
    primary_metric_name: str,
) -> Dict:
    """
    Structured ship/no-ship recommendation based on:
    1. Primary metric: statistically significant positive lift?
    2. Guardrails: any statistically significant negative impact?

    Returns a decision dict with rationale.
    """
    primary = next((r for r in primary_results if primary_metric_name in r["metric"]), None)
    if primary is None:
        return {"decision": "UNCLEAR", "rationale": "Primary metric not found in results."}

    primary_sig_positive = (
        primary.get("adjusted_significant", primary.get("significant", False)) and
        primary["direction"] == "positive"
    )

    guardrail_failures = [
        r for r in guardrail_results
        if r.get("adjusted_significant", r.get("significant", False)) and
        r["direction"] == "negative"
    ]

    if primary_sig_positive and not guardrail_failures:
        decision   = "SHIP ✅"
        confidence = "HIGH"
        rationale  = (
            f"Primary metric ({primary['metric']}) shows statistically significant lift of "
            f"{primary['rel_diff_pct']:+.2f}% (p={primary['p_value']:.4f}). "
            f"No guardrail metrics were significantly harmed."
        )
    elif primary_sig_positive and guardrail_failures:
        decision   = "HOLD ⚠️"
        confidence = "MEDIUM"
        failed = ", ".join([r["metric"] for r in guardrail_failures])
        rationale  = (
            f"Primary metric shows positive lift but guardrail(s) failed: {failed}. "
            f"Requires further investigation before shipping."
        )
    elif not primary_sig_positive and not guardrail_failures:
        decision   = "NO SHIP ❌"
        confidence = "HIGH"
        rationale  = (
            f"Primary metric ({primary['metric']}) did not show statistically significant lift "
            f"(p={primary['p_value']:.4f}, lift={primary['rel_diff_pct']:+.2f}%). "
            f"Insufficient evidence to ship."
        )
    else:
        decision   = "NO SHIP ❌"
        confidence = "HIGH"
        failed = ", ".join([r["metric"] for r in guardrail_failures])
        rationale  = (
            f"Primary metric not significant AND guardrail(s) harmed: {failed}. "
            f"Do not ship."
        )

    return {
        "decision":           decision,
        "confidence":         confidence,
        "rationale":          rationale,
        "primary_lift_pct":   primary["rel_diff_pct"],
        "primary_p_value":    primary["p_value"],
        "guardrail_failures": [r["metric"] for r in guardrail_failures],
        "n_guardrails_tested": len(guardrail_results),
    }
