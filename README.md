# A/B Test Analysis — Feed Ranking Algorithm Change

**Author:** Denzel C. Seals | Senior Data Analyst  
**Domain:** Product Analytics — Experimentation & Causal Inference  
**Stack:** Python · Pandas · SciPy · Plotly · Streamlit  
**Scenario:** Simulated Meta-style feed ranking A/B test across 500K users

---

## Overview

This project replicates the end-to-end workflow a Product Data Scientist at a
company like Meta would follow when analyzing a feed ranking experiment:

1. **Experiment design** — sample size, power calculation, MDE, randomization
2. **Sanity checks** — SRM detection, pre-experiment metric validation
3. **Statistical analysis** — t-tests, z-tests, confidence intervals
4. **Guardrail monitoring** — ensuring the treatment didn't hurt adjacent metrics
5. **Novelty effect detection** — separating real lift from short-term curiosity spikes
6. **Ship / No-Ship recommendation** — structured decision memo

---

## Business Context

A product team proposes re-ranking the Feed to surface **more content from close friends
and less from Pages/brands**, hypothesizing this increases meaningful engagement
(comments + shares) without hurting session time.

**Primary metric:** Comments + Shares per session (Meaningful Engagement Rate)  
**Guardrail metrics:** Session length, Ad impression count, Day-7 retention  
**North star:** Daily Active Users (DAU) — monitored but not directly tested

---

## Project Structure

```
ab-test-analysis/
│
├── data/
│   ├── raw/                        # Generated synthetic experiment data
│   └── processed/                  # Aggregated daily metrics, user-level results
│
├── src/
│   ├── data_generator.py           # Synthetic experiment data generator
│   ├── experiment_design.py        # Sample size, power, MDE calculations
│   ├── sanity_checks.py            # SRM test, pre-period AA test
│   └── analysis.py                 # Statistical tests, confidence intervals, segmentation
│
├── notebooks/
│   └── ab_test_analysis.ipynb      # Full experiment analysis narrative
│
├── app/
│   └── streamlit_app.py            # Interactive ship/no-ship decision simulator
│
├── outputs/
│   └── charts/                     # Exported chart PNGs
│
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/DSeals12/ab-test-analysis.git
cd ab-test-analysis
pip install -r requirements.txt

# Generate data and run notebook
jupyter notebook notebooks/ab_test_analysis.ipynb

# Launch Streamlit app
streamlit run app/streamlit_app.py
```

---

## Key Concepts Demonstrated

| Concept | Where |
|---|---|
| Power analysis & sample size calculation | `src/experiment_design.py` + Section 2 |
| Sample Ratio Mismatch (SRM) detection | `src/sanity_checks.py` + Section 3 |
| A/A test validation | `src/sanity_checks.py` + Section 3 |
| Two-sample t-test & z-test | `src/analysis.py` + Section 4 |
| Delta method for ratio metrics | `src/analysis.py` + Section 4 |
| Confidence interval interpretation | Section 4 |
| Novelty effect detection | Section 5 |
| Multiple testing correction (Bonferroni) | Section 4 |
| Segment analysis (new vs. returning users) | Section 6 |
| Ship / No-Ship decision framework | Section 7 + Streamlit app |

---

## Skills Demonstrated

- End-to-end A/B test design and analysis
- Statistical hypothesis testing (t-test, chi-square, z-test)
- Power analysis and minimum detectable effect (MDE) calculation
- Sample Ratio Mismatch (SRM) detection — a critical production sanity check
- Novelty effect identification via day-by-day metric time series
- Guardrail metric monitoring and multiple testing correction
- Structured ship/no-ship recommendation with quantified uncertainty

📊 [View rendered notebook with charts](https://nbviewer.org/github/DSeals12/ab-test-analysis/blob/main/notebooks/ab_test_analysis.ipynb)
