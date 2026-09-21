"""
data_generator.py
-----------------
Generates a synthetic A/B test dataset simulating a Meta-style
feed ranking experiment across 500K users over 14 days.

Experiment:
  Control  (50%): Current feed ranking algorithm
  Treatment (50%): New ranking — surfaces more friends content

Metrics generated per user per day:
  - sessions           : number of app opens
  - session_length_sec : average session length in seconds
  - comments           : number of comments left
  - shares             : number of shares
  - likes              : number of likes
  - ad_impressions     : number of ads seen
  - d7_retained        : whether user returned on day 7 (binary)

Usage:
    python src/data_generator.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
np.random.seed(SEED)

# ── Experiment Config ─────────────────────────────────────────────────────────

CONFIG = {
    "n_users":          500_000,
    "experiment_days":  14,
    "treatment_split":  0.50,

    # User segments
    "segments": {
        "new_user":       {"weight": 0.20, "base_sessions": 1.2,  "base_engagement": 0.8},
        "casual":         {"weight": 0.45, "base_sessions": 1.8,  "base_engagement": 1.0},
        "power_user":     {"weight": 0.35, "base_sessions": 3.2,  "base_engagement": 1.6},
    },

    # Control baseline metrics (per user per day)
    "control": {
        "sessions_mu":          2.1,
        "session_length_mu":    420,   # seconds
        "session_length_sd":    180,
        "likes_per_session":    3.2,
        "comments_per_session": 0.45,
        "shares_per_session":   0.18,
        "ad_imp_per_session":   4.8,
        "d7_retention":         0.62,
    },

    # Treatment effect (true underlying lift — what we're trying to detect)
    "treatment_effect": {
        "comments_lift":        0.08,   # +8% comments
        "shares_lift":          0.06,   # +6% shares
        "likes_lift":          -0.02,   # -2% likes (replaced by deeper engagement)
        "session_length_lift": -0.01,   # -1% session length (guardrail — small negative)
        "ad_imp_lift":         -0.03,   # -3% ad impressions (guardrail — negative)
        "d7_retention_lift":    0.005,  # +0.5pp retention
        "sessions_lift":        0.00,   # No change in session count
    },

    # Novelty effect: decays over first 5 days
    "novelty": {
        "peak_lift":   0.04,   # Extra 4% engagement on day 1
        "decay_days":  5,      # Gone by day 5
    },
}


def novelty_factor(day: int, config: dict) -> float:
    """Additive novelty lift that decays linearly to zero."""
    if day >= config["novelty"]["decay_days"]:
        return 0.0
    return config["novelty"]["peak_lift"] * (1 - day / config["novelty"]["decay_days"])


def generate_users(config: dict) -> pd.DataFrame:
    """Assign users to segments and treatment/control."""
    n = config["n_users"]
    segments = list(config["segments"].keys())
    weights  = [config["segments"][s]["weight"] for s in segments]

    users = pd.DataFrame({
        "user_id":   range(1, n + 1),
        "segment":   np.random.choice(segments, size=n, p=weights),
        "variant":   np.random.choice(
            ["control", "treatment"],
            size=n,
            p=[1 - config["treatment_split"], config["treatment_split"]],
        ),
    })
    return users


def generate_daily_metrics(users: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Generate one row per user per day with all engagement metrics."""
    ctrl  = config["control"]
    treat = config["treatment_effect"]
    records = []

    for day in range(config["experiment_days"]):
        novelty = novelty_factor(day, config)

        for segment, seg_cfg in config["segments"].items():
            seg_users = users[users["segment"] == segment]
            n_seg = len(seg_users)
            base  = seg_cfg["base_engagement"]

            for variant in ["control", "treatment"]:
                variant_mask = seg_users["variant"] == variant
                n_var = variant_mask.sum()
                if n_var == 0:
                    continue

                is_treatment = (variant == "treatment")
                nov = novelty if is_treatment else 0.0

                # Sessions
                sessions = np.random.poisson(
                    ctrl["sessions_mu"] * seg_cfg["base_sessions"] *
                    (1 + treat["sessions_lift"] * is_treatment),
                    size=n_var,
                ).clip(0)

                # Session length
                sl_mean = ctrl["session_length_mu"] * base * (
                    1 + treat["session_length_lift"] * is_treatment
                )
                session_length = np.random.normal(sl_mean, ctrl["session_length_sd"], n_var).clip(30)

                # Engagement per session
                comments = np.random.poisson(
                    ctrl["comments_per_session"] * base *
                    (1 + treat["comments_lift"] * is_treatment + nov) *
                    np.maximum(sessions, 1),
                    size=n_var,
                )
                shares = np.random.poisson(
                    ctrl["shares_per_session"] * base *
                    (1 + treat["shares_lift"] * is_treatment + nov) *
                    np.maximum(sessions, 1),
                    size=n_var,
                )
                likes = np.random.poisson(
                    ctrl["likes_per_session"] * base *
                    (1 + treat["likes_lift"] * is_treatment) *
                    np.maximum(sessions, 1),
                    size=n_var,
                )
                ad_imp = np.random.poisson(
                    ctrl["ad_imp_per_session"] * base *
                    (1 + treat["ad_imp_lift"] * is_treatment) *
                    np.maximum(sessions, 1),
                    size=n_var,
                )

                # D7 retention (only computed on day 0)
                d7 = np.random.binomial(
                    1,
                    ctrl["d7_retention"] * (1 + treat["d7_retention_lift"] * is_treatment),
                    size=n_var,
                ) if day == 0 else np.zeros(n_var, dtype=int)

                user_ids = seg_users[variant_mask]["user_id"].values

                for i in range(n_var):
                    records.append({
                        "user_id":           int(user_ids[i]),
                        "day":               day,
                        "variant":           variant,
                        "segment":           segment,
                        "sessions":          int(sessions[i]),
                        "session_length_sec":round(float(session_length[i]), 1),
                        "comments":          int(comments[i]),
                        "shares":            int(shares[i]),
                        "likes":             int(likes[i]),
                        "ad_impressions":    int(ad_imp[i]),
                        "d7_retained":       int(d7[i]),
                    })

        if day % 3 == 0:
            print(f"  Generated day {day + 1}/{config['experiment_days']}...")

    return pd.DataFrame(records)


def main():
    print("Generating users...")
    users = generate_users(CONFIG)
    print(f"  {len(users):,} users — "
          f"{(users['variant']=='treatment').sum():,} treatment, "
          f"{(users['variant']=='control').sum():,} control")

    print("Generating daily metrics...")
    df = generate_daily_metrics(users, CONFIG)

    raw_path = Path("data/raw/experiment_data.parquet")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_path, index=False)

    print(f"\nSaved: {raw_path}  ({len(df):,} rows)")
    print("Done. Open notebooks/ab_test_analysis.ipynb to run the analysis.")


if __name__ == "__main__":
    main()
