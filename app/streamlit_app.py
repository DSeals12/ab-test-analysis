"""
streamlit_app.py
----------------
Interactive A/B Test Ship / No-Ship Decision Simulator

Launch: streamlit run app/streamlit_app.py
"""

import sys
sys.path.insert(0, ".")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from src.experiment_design import required_sample_size, mde_for_sample
from src.analysis import test_metric, test_proportion, ship_decision, bonferroni_correction
from src.sanity_checks import check_srm, srm_summary_df

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="A/B Test Analyzer",
    page_icon="🧪",
    layout="wide",
)

BLUE  = "#2E86AB"
RED   = "#C73E1D"
GREEN = "#44BBA4"
AMBER = "#F18F01"
GRAY  = "#AAAAAA"

st.title("🧪 A/B Test Ship / No-Ship Simulator")
st.markdown(
    "**Scenario:** Feed Ranking Algorithm Change · "
    "**Framework:** Meta-style Product Experimentation · "
    "**Author:** Denzel C. Seals"
)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "⚙️ Experiment Design",
    "🔍 Sanity Checks",
    "📊 Results Analysis",
    "🚦 Ship Decision",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: EXPERIMENT DESIGN
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Pre-Experiment Power Analysis")
    st.markdown(
        "Define the experiment parameters before launch. "
        "This determines how long the experiment needs to run and what effects you can detect."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        baseline_mean = st.number_input("Baseline Engagement Rate (comments+shares per session)", value=0.63, step=0.01, format="%.3f")
        baseline_std  = st.number_input("Baseline Std Dev", value=1.20, step=0.05, format="%.2f")
    with c2:
        mde_pct       = st.slider("Minimum Detectable Effect (%)", 1, 20, 5) / 100
        power         = st.slider("Statistical Power (%)", 70, 95, 80) / 100
    with c3:
        alpha         = st.selectbox("Significance Level (α)", [0.05, 0.01, 0.10], index=0)
        daily_users   = st.number_input("Daily Eligible Users", value=500_000, step=10_000)

    n_req  = required_sample_size(baseline_mean, baseline_std, mde_pct, alpha, power)
    days   = n_req / (daily_users * 0.5)
    mde_abs = baseline_mean * mde_pct

    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Required N / Variant", f"{n_req:,}")
    m2.metric("Total N Required",     f"{n_req*2:,}")
    m3.metric("MDE (absolute)",        f"{mde_abs:.4f}")
    m4.metric("Estimated Runtime",     f"{days:.1f} days")

    st.markdown("---")
    st.markdown("**What these numbers mean:**")
    st.info(
        f"To detect a **{mde_pct:.0%} relative lift** in engagement rate with **{power:.0%} power** "
        f"at **α={alpha}**, you need **{n_req:,} users per variant** ({n_req*2:,} total). "
        f"At {daily_users:,.0f} daily users with a 50/50 split, this experiment should run for "
        f"approximately **{days:.1f} days**."
    )

    # MDE curve
    sample_sizes = np.linspace(10_000, 500_000, 100)
    mdes = [mde_for_sample(baseline_mean, baseline_std, int(n), alpha, power) * 100 for n in sample_sizes]

    fig = go.Figure(go.Scatter(x=sample_sizes/1000, y=mdes, mode="lines", line=dict(color=BLUE, width=2)))
    fig.add_vline(x=n_req/1000, line_dash="dash", line_color=AMBER,
                  annotation_text=f"Required N: {n_req:,}", annotation_font_color=AMBER)
    fig.update_layout(
        title="MDE vs. Sample Size per Variant",
        xaxis_title="Sample Size per Variant (thousands)",
        yaxis_title="Minimum Detectable Effect (%)",
        yaxis_ticksuffix="%",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12), height=360,
        margin=dict(l=60, r=40, t=50, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEE")
    fig.update_yaxes(showgrid=True, gridcolor="#EEE")
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: SANITY CHECKS
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Experiment Sanity Checks")
    st.markdown(
        "Before trusting any results, these checks must pass. "
        "A failed SRM check means the experiment is **invalid regardless of the p-value**."
    )

    st.markdown("### Sample Ratio Mismatch (SRM) Check")
    col1, col2 = st.columns(2)
    with col1:
        n_ctrl_srm  = st.number_input("Control users observed", value=249_850, step=100)
        n_treat_srm = st.number_input("Treatment users observed", value=250_150, step=100)
    with col2:
        expected_split = st.slider("Expected treatment split (%)", 40, 60, 50) / 100
        srm_alpha      = st.selectbox("SRM α threshold", [0.01, 0.001, 0.05], index=0)

    srm = check_srm(int(n_ctrl_srm), int(n_treat_srm), expected_split, srm_alpha)

    if srm["srm_detected"]:
        st.error(f"**{srm['verdict']}**")
    else:
        st.success(f"**{srm['verdict']}**")

    st.dataframe(srm_summary_df(srm), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### What is SRM and why does it matter?")
    st.info(
        "**Sample Ratio Mismatch (SRM)** occurs when the observed traffic split differs "
        "significantly from the intended split. This typically indicates a bug in the "
        "randomization layer — e.g. a logging issue, a filter applied after randomization, "
        "or a bot filtering that hits one variant more than another. "
        "Even a 0.5% imbalance can bias your results because the groups may differ "
        "systematically (e.g. by device type or user vintage)."
    )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: RESULTS ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Metric Results")
    st.markdown("Input your observed experiment results to run statistical tests.")

    st.markdown("### Primary Metric — Meaningful Engagement Rate")
    pc1, pc2 = st.columns(2)
    with pc1:
        ctrl_eng_mean  = st.number_input("Control engagement rate (mean)", value=0.6300, format="%.4f")
        ctrl_eng_std   = st.number_input("Control std dev", value=1.20, format="%.2f")
        ctrl_n         = st.number_input("Control N", value=249850, step=1000)
    with pc2:
        treat_eng_mean = st.number_input("Treatment engagement rate (mean)", value=0.6790, format="%.4f")
        treat_eng_std  = st.number_input("Treatment std dev", value=1.22, format="%.2f")
        treat_n        = st.number_input("Treatment N", value=250150, step=1000)

    ctrl_eng  = np.random.normal(ctrl_eng_mean,  ctrl_eng_std,  int(ctrl_n))
    treat_eng = np.random.normal(treat_eng_mean, treat_eng_std, int(treat_n))
    primary_result = test_metric(pd.Series(ctrl_eng), pd.Series(treat_eng), "Meaningful Engagement Rate", alpha=0.05)

    st.markdown("---")
    st.markdown("### Guardrail Metrics")

    gc1, gc2, gc3 = st.columns(3)
    with gc1:
        ctrl_sl  = st.number_input("Control session length (sec)", value=420.0, format="%.1f")
        treat_sl = st.number_input("Treatment session length (sec)", value=415.8, format="%.1f")
        sl_std   = st.number_input("Session length std dev", value=180.0, format="%.1f")
    with gc2:
        ctrl_ad  = st.number_input("Control ad impressions/session", value=4.80, format="%.2f")
        treat_ad = st.number_input("Treatment ad impressions/session", value=4.66, format="%.2f")
        ad_std   = st.number_input("Ad impressions std dev", value=2.10, format="%.2f")
    with gc3:
        ctrl_ret_rate  = st.number_input("Control D7 retention rate", value=0.620, format="%.3f")
        treat_ret_rate = st.number_input("Treatment D7 retention rate", value=0.623, format="%.3f")

    ctrl_sl_s  = np.random.normal(ctrl_sl,  sl_std, int(ctrl_n))
    treat_sl_s = np.random.normal(treat_sl, sl_std, int(treat_n))
    ctrl_ad_s  = np.random.normal(ctrl_ad,  ad_std, int(ctrl_n))
    treat_ad_s = np.random.normal(treat_ad, ad_std, int(treat_n))

    sl_result  = test_metric(pd.Series(ctrl_sl_s), pd.Series(treat_sl_s), "Session Length (sec)", alpha=0.05)
    ad_result  = test_metric(pd.Series(ctrl_ad_s), pd.Series(treat_ad_s), "Ad Impressions / Session", alpha=0.05)
    ret_result = test_proportion(
        int(ctrl_n), int(treat_n),
        int(ctrl_ret_rate * ctrl_n), int(treat_ret_rate * treat_n),
        "D7 Retention", alpha=0.05,
    )

    guardrail_results = bonferroni_correction([sl_result, ad_result, ret_result])

    # Results table
    st.markdown("---")
    all_results = bonferroni_correction([primary_result] + [sl_result, ad_result, ret_result])

    rows = []
    for r in all_results:
        sig = r.get("adjusted_significant", r.get("significant", False))
        rows.append({
            "Metric":       r["metric"],
            "Control":      f"{r['ctrl_mean']:.4f}",
            "Treatment":    f"{r['treat_mean']:.4f}",
            "Lift":         f"{r['rel_diff_pct']:+.2f}%",
            "p-value":      f"{r['p_value']:.5f}",
            "Significant":  "✅ Yes" if sig else "❌ No",
            "Type":         "Primary" if r["metric"] == "Meaningful Engagement Rate" else "Guardrail",
        })
    results_df = pd.DataFrame(rows)
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    # CI chart
    st.markdown("### Confidence Intervals")
    fig_ci = go.Figure()
    for r in all_results:
        color = GREEN if r["rel_diff_pct"] > 0 else RED
        fig_ci.add_trace(go.Scatter(
            x=[r["ci_low"] / r["ctrl_mean"] * 100 if r["ctrl_mean"] else 0,
               r["ci_high"] / r["ctrl_mean"] * 100 if r["ctrl_mean"] else 0],
            y=[r["metric"], r["metric"]],
            mode="lines", line=dict(color=color, width=3),
            showlegend=False,
        ))
        fig_ci.add_trace(go.Scatter(
            x=[r["rel_diff_pct"]],
            y=[r["metric"]],
            mode="markers", marker=dict(color=color, size=10),
            showlegend=False,
        ))
    fig_ci.add_vline(x=0, line_dash="dash", line_color=GRAY)
    fig_ci.update_layout(
        title="Treatment Effect with 95% Confidence Intervals",
        xaxis_title="Relative Lift (%)", xaxis_ticksuffix="%",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12), height=300,
        margin=dict(l=220, r=40, t=50, b=50),
    )
    st.plotly_chart(fig_ci, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: SHIP DECISION
# ═════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Ship / No-Ship Recommendation")

    primary_results_list  = bonferroni_correction([primary_result])
    guardrail_results_list = bonferroni_correction([sl_result, ad_result, ret_result])
    decision = ship_decision(primary_results_list, guardrail_results_list, "Meaningful Engagement Rate")

    if "SHIP ✅" in decision["decision"]:
        st.success(f"## {decision['decision']}")
    elif "HOLD" in decision["decision"]:
        st.warning(f"## {decision['decision']}")
    else:
        st.error(f"## {decision['decision']}")

    st.markdown(f"**Confidence:** {decision['confidence']}")
    st.markdown(f"**Rationale:** {decision['rationale']}")

    st.markdown("---")
    st.markdown("### Decision Memo")
    st.markdown(f"""
**Experiment:** Feed Ranking Algorithm Change  
**Date:** 2024  
**Analyst:** Denzel C. Seals

**Primary Metric Result:**  
Meaningful Engagement Rate (comments + shares per session) increased by 
**{decision['primary_lift_pct']:+.2f}%** (p={decision['primary_p_value']:.5f}).

**Guardrail Assessment:**  
Tested {decision['n_guardrails_tested']} guardrail metrics with Bonferroni correction.  
{"No guardrails were significantly harmed." if not decision['guardrail_failures']
 else f"⚠️ The following guardrails showed significant negative impact: {', '.join(decision['guardrail_failures'])}"}

**Recommendation:** {decision['decision']}  
{decision['rationale']}

**Next Steps:**
- {"Roll out to 100% of eligible users and monitor guardrails for 7 days post-launch" if "SHIP" in decision["decision"] and "HOLD" not in decision["decision"] else "Investigate guardrail failures before proceeding — consider a targeted fix or reduced rollout"}
- Set up long-term holdout (5% control) to measure 30-day retention impact
- Monitor ad revenue impact at portfolio level (not just per-session)
""")

    st.markdown("---")
    st.caption("Built by Denzel C. Seals · Meta-style A/B Test Framework · github.com/DSeals12")
