"""
experiment_design.py
--------------------
Pre-experiment design calculations:
  - Minimum Detectable Effect (MDE)
  - Required sample size
  - Statistical power
  - Expected experiment runtime

These are the calculations a DS does BEFORE launching an experiment.
At Meta, shipping a test without a power analysis is a red flag.

Usage:
    from src.experiment_design import power_analysis, mde_for_sample, experiment_brief
"""

import numpy as np
from scipy import stats
from typing import Tuple
import pandas as pd


# ── Core Formulas ─────────────────────────────────────────────────────────────

def required_sample_size(
    baseline_mean: float,
    baseline_std: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    """
    Calculate required sample size per variant for a two-sample t-test.

    Parameters
    ----------
    baseline_mean  : control group mean
    baseline_std   : control group standard deviation
    mde_relative   : minimum detectable effect as fraction of baseline (e.g. 0.05 = 5%)
    alpha          : significance level (Type I error rate), default 0.05
    power          : desired power (1 - Type II error rate), default 0.80
    two_sided      : whether to use a two-sided test

    Returns
    -------
    n per variant (integer)
    """
    mde_absolute = baseline_mean * mde_relative
    alpha_adj    = alpha / 2 if two_sided else alpha

    z_alpha = stats.norm.ppf(1 - alpha_adj)
    z_beta  = stats.norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) ** 2) * (baseline_std ** 2) / (mde_absolute ** 2)
    return int(np.ceil(n))


def achieved_power(
    baseline_mean: float,
    baseline_std: float,
    observed_effect: float,
    n_per_variant: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """
    Calculate the statistical power achieved given an observed effect and sample size.
    Useful for post-hoc power analysis.
    """
    alpha_adj = alpha / 2 if two_sided else alpha
    z_alpha   = stats.norm.ppf(1 - alpha_adj)
    se        = baseline_std * np.sqrt(2 / n_per_variant)
    z_power   = abs(observed_effect) / se - z_alpha
    return float(stats.norm.cdf(z_power))


def mde_for_sample(
    baseline_mean: float,
    baseline_std: float,
    n_per_variant: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    """
    Given a fixed sample size, what's the smallest effect we can reliably detect?
    Returns MDE as a fraction of baseline mean.
    """
    alpha_adj = alpha / 2 if two_sided else alpha
    z_alpha   = stats.norm.ppf(1 - alpha_adj)
    z_beta    = stats.norm.ppf(power)
    se        = baseline_std * np.sqrt(2 / n_per_variant)
    mde_abs   = (z_alpha + z_beta) * se
    return mde_abs / baseline_mean


def runtime_days(
    n_required: int,
    daily_eligible_users: int,
    treatment_split: float = 0.50,
) -> float:
    """
    How many days does the experiment need to run to accumulate enough users?
    """
    users_per_day_per_variant = daily_eligible_users * treatment_split
    return n_required / users_per_day_per_variant


# ── Experiment Brief ──────────────────────────────────────────────────────────

def experiment_brief(
    metric_name: str,
    baseline_mean: float,
    baseline_std: float,
    mde_relative: float,
    daily_users: int,
    alpha: float = 0.05,
    power: float = 0.80,
    treatment_split: float = 0.50,
) -> pd.DataFrame:
    """
    Generate a full pre-experiment design brief for one metric.
    This is what you'd put in an experiment design doc before launch.
    """
    n_req  = required_sample_size(baseline_mean, baseline_std, mde_relative, alpha, power)
    days   = runtime_days(n_req, daily_users, treatment_split)
    mde_abs = baseline_mean * mde_relative

    rows = [
        ("Metric",                     metric_name),
        ("Baseline Mean",              f"{baseline_mean:.4f}"),
        ("Baseline Std Dev",           f"{baseline_std:.4f}"),
        ("MDE (relative)",             f"{mde_relative:.1%}"),
        ("MDE (absolute)",             f"{mde_abs:.4f}"),
        ("Significance Level (α)",     f"{alpha:.2f}"),
        ("Statistical Power (1−β)",    f"{power:.0%}"),
        ("Required N per Variant",     f"{n_req:,}"),
        ("Total N Required",           f"{n_req * 2:,}"),
        ("Daily Eligible Users",       f"{daily_users:,}"),
        ("Treatment Split",            f"{treatment_split:.0%} / {1-treatment_split:.0%}"),
        ("Estimated Runtime",          f"{days:.1f} days"),
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


# ── Multi-Metric Design Table ──────────────────────────────────────────────────

def design_table(metrics: list, daily_users: int, alpha: float = 0.05, power: float = 0.80) -> pd.DataFrame:
    """
    Build a design summary across all experiment metrics.

    metrics: list of dicts with keys: name, baseline_mean, baseline_std, mde_relative
    """
    rows = []
    for m in metrics:
        n = required_sample_size(m["baseline_mean"], m["baseline_std"], m["mde_relative"], alpha, power)
        days = runtime_days(n, daily_users)
        rows.append({
            "metric":          m["name"],
            "baseline_mean":   round(m["baseline_mean"], 4),
            "mde_relative":    f"{m['mde_relative']:.1%}",
            "mde_absolute":    round(m["baseline_mean"] * m["mde_relative"], 4),
            "n_per_variant":   f"{n:,}",
            "runtime_days":    round(days, 1),
        })
    df = pd.DataFrame(rows)
    # Primary metric drives the runtime (longest needed)
    df["is_binding"] = df["runtime_days"] == df["runtime_days"].max()
    return df
