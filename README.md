# Bachelor Thesis

This repository contains the final thesis, Python replication code, and
aggregate empirical results for Junpeng Wang's bachelor thesis.

## Repository contents

- `Thesis_Junpeng_Wang.pdf`: final bachelor thesis
- `thesis_analysis.py`: sample construction and empirical analysis
- `requirements.txt`: Python dependencies
- `DATA_REQUIREMENTS.md`: instructions for supplying licensed source data
- `results/`: aggregate regression tables and processing logs

The raw Compustat extract and firm-level processed dataset are intentionally
excluded because their licences do not permit public redistribution.

## Reproduce the analysis

Use Python 3.9 or newer. From the repository root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 thesis_analysis.py
```

Before running the analysis, obtain authorised access to the source data and
save the extract as `data.csv` in the repository root. See
`DATA_REQUIREMENTS.md` for the required columns. Git ignores this file so it
cannot be committed accidentally.

## Scope

The code covers sample construction, variable construction, target-leverage
estimation, H1 peer benchmark models, H2 acquisition-adjustment models, and H3
acquisition-decision models. The later resit robustness analysis is not part of
this repository.

