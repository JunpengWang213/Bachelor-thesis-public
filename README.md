# Leverage Benchmarks, Capital Structure Adjustment, and Acquisition Activity

Replication materials for my BSc Economics and Business Economics thesis in
the Finance specialisation at the University of Amsterdam.

The study examines how peer and target leverage benchmarks are associated with
firms' capital structure adjustment and acquisition activity. The analysis uses
Compustat North America data for U.S.-listed industrial firms from 2000 to 2024.

## Research question

How are peer and target leverage benchmarks associated with firms' capital
structure adjustment and acquisition activity?

## Hypotheses

| Hypothesis | Prediction |
| --- | --- |
| H1 | Firm leverage is positively associated with peer firm leverage. |
| H2 | Firms adjust their leverage deviations following acquisition activity. |
| H3 | Firms with higher leverage deficits undertake less acquisition activity in the future. |

Peer leverage is the leave-one-out average leverage of other firms in the same
three-digit SIC industry and fiscal year. Target leverage is estimated annually
from firm characteristics. The leverage deficit is defined as actual market
leverage minus estimated target leverage.

## Data and empirical approach

- Annual Compustat North America fundamentals, 2000-2024
- U.S.-incorporated industrial firms, excluding financial firms and utilities
- Base sample of 114,702 firm-year observations across 12,900 firms
- Acquisition expenditure scaled by total assets as the acquisition-intensity proxy
- Linear panel-style regressions with year and, where applicable, SIC3 fixed effects
- Standard errors clustered at the firm level

The raw Compustat extract and firm-level processed dataset are excluded because
their licences do not permit public redistribution.

## Main results

| Test | Main estimate | Interpretation |
| --- | ---: | --- |
| H1: Peer leverage | `0.0656***` (t = 7.16) | Peer market leverage remains positively associated with firm market leverage in the fully controlled model. The thesis shows that this result is sensitive to the leverage definition. |
| H2: Three-year adjustment | `-0.9455***` (t = -82.40) | Within acquisition firm-years, larger initial leverage deviations are followed by larger reductions in those deviations. The result should be interpreted with caution because estimated targets, mean reversion, and measurement error can affect the coefficient. |
| H3: Future acquisition intensity | `-0.0254***` (t = -9.52) | Higher leverage deficits predict lower acquisition intensity in the following fiscal year. |

`***` denotes statistical significance at the 1% level. The results are
conditional associations and do not establish causal effects.

## Repository contents

- `Thesis_Junpeng_Wang.pdf`: final thesis
- `thesis_analysis.py`: sample construction, variable construction, and empirical analysis
- `requirements.txt`: Python dependencies
- `DATA_REQUIREMENTS.md`: required fields for an authorised Compustat extract
- `results/`: aggregate regression tables and processing logs

## Reproduce the analysis

Use Python 3.9 or newer. From the repository root, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 thesis_analysis.py
```

Before running the script, obtain authorised access to the source data and save
the extract as `data.csv` in the repository root. The required columns are
listed in `DATA_REQUIREMENTS.md`. Git ignores this file to prevent accidental
publication.

The script reproduces the main sample construction and the H1-H3 regression
tables. The additional robustness analyses reported in the thesis are outside
the scope of the public replication code.
