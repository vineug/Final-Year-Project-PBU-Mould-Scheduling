# Intelligent Mould Scheduling for Precast Panel Production

A Constraint Programming and Machine Learning Framework for optimising the scheduling of L-shaped precast panel casting in Prefabricated Bathroom Unit (PBU) manufacturing.

## Overview

This framework combines:
- **Batching heuristic** — groups panels into geometrically feasible mould cycles respecting panel handedness, spacing, and ceiling constraints
- **CP-SAT sequencing** — optimises cycle ordering on each mould to minimise production makespan using Google OR-Tools
- **ML changeover prediction** — Gradient Boosting model trained on synthetic data, integrated as alternative arc weights in the CP-SAT solver
- **Interactive dashboard** — Streamlit-based interface for production planners

## Repository Structure

```
├── README.md
├── requirements.txt
├── app.py                      # Streamlit dashboard
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
└── data/
    └── panels.csv              # Panel dataset (25 PBUs, 75 panels)
```

## Running the Dashboard

### Live Demo
The dashboard is deployed at: (https://pbu-mould-scheduling-gkd2j5u5ukru565fvfbg5x.streamlit.app)

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Key Results

- 25 PBUs (75 panels) scheduled across 6 moulds in 23 cycles
- Baseline makespan: 43.5 hours
- CP-SAT optimised: 40.0 hours (8.0% improvement)

## Technology Stack

| Component | Tool |
|-----------|------|
| Optimisation | Google OR-Tools CP-SAT |
| Machine Learning | scikit-learn (Gradient Boosting) |
| Dashboard | Streamlit |
| Visualisation | Plotly |
| Language | Python 3.10+ |

## Author

Final Year Project 2025/2026
