# Setup

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

try:
    from IPython.display import display
except ImportError:
    def display(value):
        print(value)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "results"
OUTPUT_FOLDER.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        "data.csv is not distributed with this public repository. "
        "Obtain the licensed Compustat data described in DATA_REQUIREMENTS.md "
        "and save it as data.csv in the repository root."
    )

# The three-year horizon is the main specification; one and five years are
# reported as alternatives.
ADJUSTMENT_HORIZONS = [1, 3, 5]

# HKW use large acquisitions. Compustat AQC scaled by assets is the closest
# available proxy without SDC deal-level data.
LARGE_ACQ_THRESHOLD = 0.20


df_raw = pd.read_csv(DATA_PATH)


# Initial data checks

print("the shape is", df_raw.shape)
display(df_raw.head(15))
print(df_raw.dtypes)

missing = pd.DataFrame({
    "missing_count": df_raw.isna().sum(),
    "missing_pct": df_raw.isna().mean()
}).sort_values("missing_pct", ascending=False)

display(missing)

print("Fiscal year range:")
print(df_raw["fyear"].min(), df_raw["fyear"].max())

if "fic" in df_raw.columns:
    print(df_raw["fic"].value_counts().head(20))

if "sic" in df_raw.columns:
    print(df_raw["sic"].head())
    print(df_raw["sic"].describe())

# Inspect recent fiscal years for incomplete coverage.
year_check = (
    df_raw
    .groupby("fyear")
    .agg(
        observations=("gvkey", "count"),
        unique_firms=("gvkey", "nunique"),
        missing_at=("at", lambda x: x.isna().mean()),
        missing_dlc=("dlc", lambda x: x.isna().mean()),
        missing_dltt=("dltt", lambda x: x.isna().mean()),
        missing_aqc=("aqc", lambda x: x.isna().mean())
    )
    .reset_index()
)

display(year_check.tail(10))


# Data preparation

processing_log = []


def log_step(step_name, data):
    processing_log.append({
        "step": step_name,
        "observations": len(data),
        "unique_firms": data["gvkey"].nunique() if "gvkey" in data.columns else np.nan
    })
    print(
        f"{step_name}: {data.shape}, "
        f"unique firms: {data['gvkey'].nunique() if 'gvkey' in data.columns else 'N/A'}"
    )


def make_peer_average(data, group_cols, value_col, output_col):
    peer_sum = data.groupby(group_cols)[value_col].transform("sum")
    peer_count = data.groupby(group_cols)[value_col].transform("count")
    data[output_col] = (peer_sum - data[value_col]) / (peer_count - 1)
    data.loc[peer_count <= 1, output_col] = np.nan
    return data


def winsorize_column(data, col, lower=0.01, upper=0.99):
    p_low = data[col].quantile(lower)
    p_high = data[col].quantile(upper)
    data[col + "_w"] = data[col].clip(lower=p_low, upper=p_high)
    return data


df_raw.columns = df_raw.columns.str.lower().str.strip()

df = df_raw.copy()
log_step("raw data", df)

print("raw shape:", df.shape)
print(df.columns.tolist())
display(df.head())

if "datadate" in df.columns:
    df["datadate"] = pd.to_datetime(df["datadate"], errors="coerce")

numeric_cols = [
    "gvkey", "cik", "naics", "naicsh", "sic", "fyear",
    "at", "ceq", "che", "dlc", "dltt", "optprcca",
    "ppent", "ebit", "oibdp", "sale", "xrd", "aqc",
    "csho", "prcc_c", "prcc_f"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

log_step("after type conversion", df)


# Sample restrictions

if "consol" in df.columns:
    df = df[df["consol"] == "C"]

if "indfmt" in df.columns:
    df = df[df["indfmt"] == "INDL"]

if "datafmt" in df.columns:
    df = df[df["datafmt"] == "STD"]

if "curcd" in df.columns:
    df = df[df["curcd"] == "USD"]

log_step("after keeping standard industrial USD records", df)

if "fic" in df.columns:
    df = df[df["fic"] == "USA"]
    log_step("after keeping U.S. firms", df)

df = df[df["fyear"].between(2000, 2024)]
log_step("after sample period restriction", df)

# Financial firms and utilities face different regulatory constraints.
df = df[~df["sic"].between(6000, 6999)]
df = df[~df["sic"].between(4900, 4999)]
log_step("after excluding financials and utilities", df)

df = df[df["at"] > 0]
log_step("after requiring positive assets", df)

core_cols = [
    "at", "dltt", "dlc", "ceq", "che", "csho", "prcc_f",
    "oibdp", "ebit", "ppent", "sale", "xrd", "aqc"
]

missing_after_restrictions = pd.DataFrame({
    "missing_count": df[core_cols].isna().sum(),
    "missing_pct": df[core_cols].isna().mean()
}).sort_values("missing_pct", ascending=False)

display(missing_after_restrictions)


# Variable construction

df["dltt"] = df["dltt"].fillna(0)
df["dlc"] = df["dlc"].fillna(0)
df["aqc"] = df["aqc"].fillna(0)

df["xrd_missing"] = df["xrd"].isna().astype(int)
df["xrd"] = df["xrd"].fillna(0)

df["total_debt"] = df["dltt"] + df["dlc"]
df["book_leverage"] = df["total_debt"] / df["at"]

df["market_equity"] = df["csho"] * df["prcc_f"]
df["market_assets"] = df["at"] - df["ceq"] + df["market_equity"]
df["market_leverage"] = df["total_debt"] / df["market_assets"]

df["size"] = np.log(df["at"])
df["profitability"] = df["oibdp"] / df["at"]
df["ebit_profitability"] = df["ebit"] / df["at"]
df["tangibility"] = df["ppent"] / df["at"]
df["cash"] = df["che"] / df["at"]
df["rd_intensity"] = df["xrd"] / df["at"]
df["market_to_book"] = df["market_assets"] / df["at"]

df["sic3"] = (df["sic"] // 10).astype(int)
df["sic2"] = (df["sic"] // 100).astype(int)

# AQC is a firm-year acquisition-expenditure proxy, not an SDC deal-level measure.
df["aqc_pos"] = df["aqc"].clip(lower=0)
df["firm_acq_intensity"] = df["aqc_pos"] / df["at"]
df["acq_dummy"] = (df["aqc_pos"] > 0).astype(int)
df["large_acq_dummy"] = (df["firm_acq_intensity"] > LARGE_ACQ_THRESHOLD).astype(int)

# Descriptive industry-year acquisition measure
df["industry_aqc"] = df.groupby(["sic3", "fyear"])["aqc_pos"].transform("sum")
df["industry_at"] = df.groupby(["sic3", "fyear"])["at"].transform("sum")
df["industry_acq_intensity"] = df["industry_aqc"] / df["industry_at"]

# Leave-one-out peer leverage within SIC3-year groups
df = make_peer_average(df, ["sic3", "fyear"], "market_leverage", "peer_leverage")

# Lagged variables and the price-based return proxy
df = df.sort_values(["gvkey", "fyear"])
df["lag_fyear"] = df.groupby("gvkey")["fyear"].shift(1)
df["lag_market_leverage"] = df.groupby("gvkey")["market_leverage"].shift(1)
df["lag_book_leverage"] = df.groupby("gvkey")["book_leverage"].shift(1)
df["lag_prcc_f"] = df.groupby("gvkey")["prcc_f"].shift(1)

# Keep lags only when the preceding observation is exactly one year earlier.
df.loc[df["fyear"] - df["lag_fyear"] != 1, "lag_market_leverage"] = np.nan
df.loc[df["fyear"] - df["lag_fyear"] != 1, "lag_book_leverage"] = np.nan
df.loc[df["fyear"] - df["lag_fyear"] != 1, "lag_prcc_f"] = np.nan

df["stock_return"] = (df["prcc_f"] - df["lag_prcc_f"]) / df["lag_prcc_f"]
df.loc[df["lag_prcc_f"] <= 0, "stock_return"] = np.nan
df.loc[df["prcc_f"] <= 0, "stock_return"] = np.nan

log_step("after variable construction", df)


# Inspect raw distributions

ratio_vars = [
    "book_leverage", "market_leverage", "market_to_book",
    "profitability", "ebit_profitability", "tangibility", "cash", "rd_intensity",
    "peer_leverage", "firm_acq_intensity", "industry_acq_intensity", "stock_return",
    "lag_market_leverage", "lag_book_leverage"
]

level_vars = [
    "at", "sale", "aqc", "total_debt", "market_assets", "market_equity"
]

check_vars = ratio_vars + level_vars

extreme_summary = df[check_vars].describe(
    percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
).T

display(extreme_summary)

print("R&D missing share:")
print(df["xrd_missing"].mean())

plt.figure(figsize=(11, 5))
df[ratio_vars].dropna().boxplot(rot=45)
plt.title("boxplots of ratio variables before cleaning")
plt.ylabel("value")
plt.show()

level_plot = df[level_vars].copy()

for col in level_vars:
    level_plot[f"log_{col}"] = np.log1p(level_plot[col].clip(lower=0))

log_level_vars = [f"log_{col}" for col in level_vars]

plt.figure(figsize=(10, 5))
level_plot[log_level_vars].dropna().boxplot(rot=45)
plt.title("boxplots of level variables before cleaning")
plt.ylabel("log(1 + value)")
plt.show()


# Clean invalid values

df_clean = df.copy()

df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

# Treat economically impossible values as missing.
df_clean.loc[df_clean["market_equity"] < 0, "market_equity"] = np.nan
df_clean.loc[df_clean["total_debt"] < 0, "total_debt"] = np.nan
df_clean.loc[df_clean["sale"] < 0, "sale"] = np.nan
df_clean.loc[df_clean["aqc"] < 0, "aqc"] = np.nan

# Recompute derived measures after cleaning their components.
df_clean["market_assets"] = df_clean["at"] - df_clean["ceq"] + df_clean["market_equity"]
df_clean.loc[df_clean["market_assets"] <= 0, "market_assets"] = np.nan

df_clean["book_leverage"] = df_clean["total_debt"] / df_clean["at"]
df_clean["market_leverage"] = df_clean["total_debt"] / df_clean["market_assets"]
df_clean["market_to_book"] = df_clean["market_assets"] / df_clean["at"]
df_clean["size"] = np.log(df_clean["at"])
df_clean["profitability"] = df_clean["oibdp"] / df_clean["at"]
df_clean["ebit_profitability"] = df_clean["ebit"] / df_clean["at"]
df_clean["tangibility"] = df_clean["ppent"] / df_clean["at"]
df_clean["cash"] = df_clean["che"] / df_clean["at"]
df_clean["rd_intensity"] = df_clean["xrd"] / df_clean["at"]

df_clean.loc[df_clean["book_leverage"] < 0, "book_leverage"] = np.nan
df_clean.loc[df_clean["market_leverage"] < 0, "market_leverage"] = np.nan
df_clean.loc[df_clean["market_to_book"] < 0, "market_to_book"] = np.nan
df_clean.loc[df_clean["tangibility"] < 0, "tangibility"] = np.nan
df_clean.loc[df_clean["cash"] < 0, "cash"] = np.nan
df_clean.loc[df_clean["rd_intensity"] < 0, "rd_intensity"] = np.nan

# Negative profitability is economically meaningful and is retained.

# Recompute acquisition measures after cleaning AQC.
df_clean["aqc_pos"] = df_clean["aqc"].clip(lower=0)
df_clean["firm_acq_intensity"] = df_clean["aqc_pos"] / df_clean["at"]
df_clean["acq_dummy"] = np.where(
    df_clean["aqc_pos"].isna(),
    np.nan,
    (df_clean["aqc_pos"] > 0).astype(float)
)
df_clean["large_acq_dummy"] = np.where(
    df_clean["firm_acq_intensity"].isna(),
    np.nan,
    (df_clean["firm_acq_intensity"] > LARGE_ACQ_THRESHOLD).astype(float)
)

df_clean.loc[df_clean["firm_acq_intensity"] < 0, "firm_acq_intensity"] = np.nan

df_clean["industry_aqc"] = df_clean.groupby(["sic3", "fyear"])["aqc_pos"].transform("sum")
df_clean["industry_at"] = df_clean.groupby(["sic3", "fyear"])["at"].transform("sum")
df_clean["industry_acq_intensity"] = df_clean["industry_aqc"] / df_clean["industry_at"]
df_clean.loc[df_clean["industry_acq_intensity"] < 0, "industry_acq_intensity"] = np.nan

df_clean = make_peer_average(df_clean, ["sic3", "fyear"], "market_leverage", "peer_leverage")
df_clean.loc[df_clean["peer_leverage"] < 0, "peer_leverage"] = np.nan

df_clean = df_clean.sort_values(["gvkey", "fyear"])
df_clean["lag_fyear"] = df_clean.groupby("gvkey")["fyear"].shift(1)
df_clean["lag_market_leverage"] = df_clean.groupby("gvkey")["market_leverage"].shift(1)
df_clean["lag_book_leverage"] = df_clean.groupby("gvkey")["book_leverage"].shift(1)
df_clean["lag_prcc_f"] = df_clean.groupby("gvkey")["prcc_f"].shift(1)

df_clean.loc[df_clean["fyear"] - df_clean["lag_fyear"] != 1, "lag_market_leverage"] = np.nan
df_clean.loc[df_clean["fyear"] - df_clean["lag_fyear"] != 1, "lag_book_leverage"] = np.nan
df_clean.loc[df_clean["fyear"] - df_clean["lag_fyear"] != 1, "lag_prcc_f"] = np.nan

df_clean["stock_return"] = (df_clean["prcc_f"] - df_clean["lag_prcc_f"]) / df_clean["lag_prcc_f"]
df_clean.loc[df_clean["lag_prcc_f"] <= 0, "stock_return"] = np.nan
df_clean.loc[df_clean["prcc_f"] <= 0, "stock_return"] = np.nan

df_clean.loc[df_clean["lag_market_leverage"] < 0, "lag_market_leverage"] = np.nan
df_clean.loc[df_clean["lag_book_leverage"] < 0, "lag_book_leverage"] = np.nan

log_step("after house cleaning", df_clean)


# Winsorization

winsor_vars = [
    "book_leverage", "market_leverage", "market_to_book",
    "profitability", "ebit_profitability", "tangibility", "cash", "rd_intensity",
    "size", "peer_leverage", "firm_acq_intensity", "industry_acq_intensity",
    "stock_return", "lag_market_leverage", "lag_book_leverage"
]

for col in winsor_vars:
    df_clean = winsorize_column(df_clean, col)

# Recompute peer leverage from winsorized firm leverage.
df_clean = make_peer_average(df_clean, ["sic3", "fyear"], "market_leverage_w", "peer_leverage_w")

# Lagged firm controls
df_clean = df_clean.sort_values(["gvkey", "fyear"])
df_clean["lag_fyear"] = df_clean.groupby("gvkey")["fyear"].shift(1)

lag_source_vars = [
    "size_w", "profitability_w", "tangibility_w", "cash_w",
    "market_to_book_w", "rd_intensity_w", "xrd_missing",
    "stock_return_w", "firm_acq_intensity_w"
]

for col in lag_source_vars:
    lag_col = "lag_" + col
    df_clean[lag_col] = df_clean.groupby("gvkey")[col].shift(1)
    df_clean.loc[df_clean["fyear"] - df_clean["lag_fyear"] != 1, lag_col] = np.nan

# Leave-one-out averages of lagged peer characteristics
peer_lag_map = {
    "lag_size_w": "peer_size_lag_w",
    "lag_profitability_w": "peer_profitability_lag_w",
    "lag_tangibility_w": "peer_tangibility_lag_w",
    "lag_market_to_book_w": "peer_market_to_book_lag_w",
    "lag_cash_w": "peer_cash_lag_w",
    "lag_rd_intensity_w": "peer_rd_intensity_lag_w"
}

for source_col, out_col in peer_lag_map.items():
    df_clean = make_peer_average(df_clean, ["sic3", "fyear"], source_col, out_col)

# Post-winsorization checks

winsor_check_vars = [col + "_w" for col in winsor_vars]

summary_after = df_clean[winsor_check_vars].describe(
    percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
).T

display(summary_after)

print("before cleaning:", df.shape)
print("after literature cleaning:", df_clean.shape)


# Inspect cleaned variables

ratio_core_vars_clean = [
    "book_leverage_w", "market_leverage_w", "market_to_book_w",
    "profitability_w", "ebit_profitability_w", "tangibility_w", "cash_w",
    "rd_intensity_w", "peer_leverage_w", "firm_acq_intensity_w",
    "industry_acq_intensity_w", "stock_return_w", "lag_market_leverage_w",
    "lag_book_leverage_w"
]

level_core_vars_clean = [
    "at", "sale", "aqc", "total_debt", "market_assets", "market_equity"
]

ratio_summary_clean = df_clean[ratio_core_vars_clean].describe(
    percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
).T

display(ratio_summary_clean)

plt.figure(figsize=(12, 5))
df_clean[ratio_core_vars_clean].dropna().boxplot(rot=45)
plt.title("boxplots of core ratio variables after cleaning")
plt.ylabel("value")
plt.show()

level_summary_clean = df_clean[level_core_vars_clean].describe(
    percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]
).T

display(level_summary_clean)

level_plot_clean = df_clean[level_core_vars_clean].copy()

for col in level_core_vars_clean:
    level_plot_clean["log_" + col] = np.log1p(level_plot_clean[col].clip(lower=0))

log_level_core_vars_clean = ["log_" + col for col in level_core_vars_clean]

plt.figure(figsize=(12, 5))
level_plot_clean[log_level_core_vars_clean].dropna().boxplot(rot=45)
plt.title("boxplots of core level variables after cleaning")
plt.ylabel("log(1 + value)")
plt.show()


# Target leverage

# Following the Uysal target-capital-structure approach, target leverage is
# estimated from firm fundamentals before the leverage deficit is calculated.
# Stock return is price-based because CRSP returns are unavailable here.

TARGET_CONTROLS = [
    "lag_size_w", "market_to_book_w", "stock_return_w",
    "rd_intensity_w", "tangibility_w", "profitability_w",
    "lag_market_leverage_w"
]

TARGET_FORMULA_FULL = (
    "market_leverage_w ~ lag_size_w + market_to_book_w + stock_return_w + "
    "rd_intensity_w + tangibility_w + profitability_w + "
    "lag_market_leverage_w + C(sic3)"
)

TARGET_FORMULA_SIMPLE = (
    "market_leverage_w ~ lag_size_w + market_to_book_w + stock_return_w + "
    "rd_intensity_w + tangibility_w + profitability_w + "
    "lag_market_leverage_w"
)


def estimate_target_leverage(data):
    data = data.copy()
    data["target_leverage"] = np.nan
    target_log = []

    for year in sorted(data["fyear"].dropna().unique()):
        year_mask = data["fyear"] == year
        train = data.loc[year_mask].dropna(
            subset=["market_leverage_w", "sic3"] + TARGET_CONTROLS
        ).copy()

        if len(train) < 50:
            target_log.append({
                "fyear": year,
                "observations": len(train),
                "formula": "skipped",
                "status": "too few observations"
            })
            continue

        # use industry fixed effects only when the annual cross-section is large enough
        formula = TARGET_FORMULA_FULL
        if len(train) <= train["sic3"].nunique() + len(TARGET_CONTROLS) + 20:
            formula = TARGET_FORMULA_SIMPLE

        try:
            model = smf.ols(formula, data=train).fit()
        except Exception as e:
            formula = TARGET_FORMULA_SIMPLE
            model = smf.ols(formula, data=train).fit()
            target_log.append({
                "fyear": year,
                "observations": len(train),
                "formula": formula,
                "status": f"full formula failed, used simple formula: {e}"
            })

        pred_data = data.loc[year_mask].dropna(subset=["sic3"] + TARGET_CONTROLS).copy()

        if len(pred_data) > 0:
            data.loc[pred_data.index, "target_leverage"] = model.predict(pred_data)

        if not any(item.get("fyear") == year for item in target_log):
            target_log.append({
                "fyear": year,
                "observations": len(train),
                "formula": formula,
                "status": "estimated"
            })

    return data, pd.DataFrame(target_log)


df_clean, target_model_log = estimate_target_leverage(df_clean)

# Keep fitted targets within the observed leverage range.
market_lev_p1 = df_clean["market_leverage_w"].quantile(0.01)
market_lev_p99 = df_clean["market_leverage_w"].quantile(0.99)
df_clean["target_leverage"] = df_clean["target_leverage"].clip(
    lower=market_lev_p1,
    upper=market_lev_p99
)

df_clean["leverage_deficit"] = df_clean["market_leverage_w"] - df_clean["target_leverage"]
df_clean = winsorize_column(df_clean, "leverage_deficit")

# Fiscal-year quartiles for the Uysal-style asymmetric specification
rank_deficit = df_clean.groupby("fyear")["leverage_deficit_w"].rank(pct=True)
df_clean["overleveraged"] = np.where(rank_deficit >= 0.75, 1, 0)
df_clean["underleveraged"] = np.where(rank_deficit <= 0.25, 1, 0)
df_clean.loc[df_clean["leverage_deficit_w"].isna(), ["overleveraged", "underleveraged"]] = np.nan

print("target leverage estimation log:")
display(target_model_log)

target_model_log.to_csv(
    f"{OUTPUT_FOLDER}/target_leverage_estimation_log.csv",
    index=False
)

# Forward variables for H2 and H3

# Merge exact calendar-year leads to avoid false leads when firms skip years.

def add_future_variable(data, var, k, output_col):
    future = data[["gvkey", "fyear", var]].drop_duplicates(
        subset=["gvkey", "fyear"],
        keep="last"
    ).copy()
    future["fyear"] = future["fyear"] - k
    future = future.rename(columns={var: output_col})
    data = data.merge(future, on=["gvkey", "fyear"], how="left")
    return data


for k in ADJUSTMENT_HORIZONS:
    df_clean = add_future_variable(
        df_clean, "leverage_deficit_w", k, f"future_leverage_deficit_{k}"
    )
    df_clean = add_future_variable(df_clean, "target_leverage", k, f"future_target_leverage_{k}")
    df_clean[f"delta_leverage_deviation_{k}"] = (
        df_clean[f"future_leverage_deficit_{k}"] - df_clean["leverage_deficit_w"]
    )
    df_clean[f"target_leverage_change_{k}"] = (
        df_clean[f"future_target_leverage_{k}"] - df_clean["target_leverage"]
    )

# Future acquisition activity
df_clean = add_future_variable(df_clean, "firm_acq_intensity_w", 1, "future_firm_acq_intensity_1")
df_clean = add_future_variable(df_clean, "acq_dummy", 1, "future_acq_dummy_1")
df_clean = add_future_variable(df_clean, "large_acq_dummy", 1, "future_large_acq_dummy_1")

log_step("after target and future variable construction", df_clean)


# Regression sample checks

h1_vars = [
    "market_leverage_w", "peer_leverage_w",
    "peer_size_lag_w", "peer_profitability_lag_w", "peer_tangibility_lag_w",
    "peer_market_to_book_lag_w",
    "lag_size_w", "lag_profitability_w", "lag_tangibility_w",
    "lag_cash_w", "lag_market_to_book_w", "lag_rd_intensity_w",
    "lag_market_leverage_w", "gvkey", "fyear", "sic3"
]

h2_vars = [
    "delta_leverage_deviation_3", "leverage_deficit_w", "firm_acq_intensity_w",
    "target_leverage_change_3", "size_w", "profitability_w", "cash_w",
    "market_to_book_w", "gvkey", "fyear", "sic3"
]

h3_vars = [
    "future_firm_acq_intensity_1", "future_acq_dummy_1", "leverage_deficit_w",
    "overleveraged", "underleveraged", "size_w", "profitability_w", "tangibility_w",
    "cash_w", "market_to_book_w", "rd_intensity_w", "xrd_missing",
    "gvkey", "fyear", "sic3"
]

sample_checks = []

for sample_name, var_list, filter_func in [
    ("H1 peer benchmark sample", h1_vars, None),
    ("H2 acquisition adjustment sample", h2_vars, lambda d: d["firm_acq_intensity_w"] > 0),
    ("H2 large acquisition adjustment sample", h2_vars, lambda d: d["large_acq_dummy"] == 1),
    ("H3 acquisition decision sample", h3_vars, None)
]:
    temp = df_clean.dropna(subset=var_list).copy()
    if filter_func is not None:
        temp = temp.loc[filter_func(temp)].copy()

    sample_checks.append({
        "sample": sample_name,
        "observations": len(temp),
        "unique_firms": temp["gvkey"].nunique(),
        "first_year": temp["fyear"].min(),
        "last_year": temp["fyear"].max()
    })

sample_check_table = pd.DataFrame(sample_checks)
display(sample_check_table)

sample_check_table.to_csv(
    f"{OUTPUT_FOLDER}/final_regression_sample_check.csv",
    index=False
)

processing_table = pd.DataFrame(processing_log)
processing_table["observations_removed"] = (
    processing_table["observations"].shift(1) - processing_table["observations"]
)
processing_table.loc[0, "observations_removed"] = np.nan

display(processing_table)

processing_table.to_csv(
    f"{OUTPUT_FOLDER}/sample_selection_table.csv",
    index=False
)

print("sample selection table saved.")


# Model specifications

DATA = df_clean

# H1: peer benchmark model
H1_FIRM_CONTROLS = [
    "lag_size_w", "lag_profitability_w", "lag_tangibility_w",
    "lag_cash_w", "lag_market_to_book_w", "lag_rd_intensity_w", "lag_xrd_missing"
]

H1_PEER_CONTROLS = [
    "peer_size_lag_w", "peer_profitability_lag_w", "peer_tangibility_lag_w",
    "peer_market_to_book_lag_w"
]

H1_COMMON_SAMPLE_VARS = [
    "market_leverage_w", "peer_leverage_w", "lag_market_leverage_w",
    "gvkey", "fyear", "sic3"
] + H1_FIRM_CONTROLS + H1_PEER_CONTROLS

H1_TABLE_ROWS = {
    "Peer leverage": "peer_leverage_w",
    "Peer size, lagged": "peer_size_lag_w",
    "Peer profitability, lagged": "peer_profitability_lag_w",
    "Peer tangibility, lagged": "peer_tangibility_lag_w",
    "Peer market-to-book, lagged": "peer_market_to_book_lag_w",
    "Lagged market leverage": "lag_market_leverage_w",
    "Size, lagged": "lag_size_w",
    "Profitability, lagged": "lag_profitability_w",
    "Tangibility, lagged": "lag_tangibility_w",
    "Cash holdings, lagged": "lag_cash_w",
    "Market-to-book, lagged": "lag_market_to_book_w",
    "R&D intensity, lagged": "lag_rd_intensity_w",
    "R&D missing indicator, lagged": "lag_xrd_missing"
}

H1_MODEL_SPECS = [
    {
        "name": "H1 M1 Peer only",
        "dep": "market_leverage_w",
        "xvars": ["peer_leverage_w"],
        "controls": [],
        "common_sample_vars": H1_COMMON_SAMPLE_VARS,
        "year_fe": True,
        "industry_fe": False
    },
    {
        "name": "H1 M2 Peer + firm controls",
        "dep": "market_leverage_w",
        "xvars": ["peer_leverage_w"],
        "controls": H1_FIRM_CONTROLS,
        "common_sample_vars": H1_COMMON_SAMPLE_VARS,
        "year_fe": True,
        "industry_fe": True
    },
    {
        "name": "H1 M3 Peer + peer controls",
        "dep": "market_leverage_w",
        "xvars": ["peer_leverage_w"] + H1_PEER_CONTROLS,
        "controls": H1_FIRM_CONTROLS,
        "common_sample_vars": H1_COMMON_SAMPLE_VARS,
        "year_fe": True,
        "industry_fe": True
    },
    {
        "name": "H1 M4 Full + lagged leverage",
        "dep": "market_leverage_w",
        "xvars": ["peer_leverage_w"] + H1_PEER_CONTROLS + ["lag_market_leverage_w"],
        "controls": H1_FIRM_CONTROLS,
        "common_sample_vars": H1_COMMON_SAMPLE_VARS,
        "year_fe": True,
        "industry_fe": True
    }
]


# H2: acquisition adjustment model
H2_CONTROLS = [
    "size_w", "profitability_w", "cash_w", "market_to_book_w"
]

H2_COMMON_BASE_VARS = [
    "leverage_deficit_w", "firm_acq_intensity_w", "size_w", "profitability_w",
    "cash_w", "market_to_book_w", "gvkey", "fyear", "sic3"
]

H2_TABLE_ROWS = {
    "Leverage deviation": "leverage_deficit_w",
    "Acquisition intensity": "firm_acq_intensity_w",
    "Target leverage change": "target_leverage_change_3",
    "Size": "size_w",
    "Profitability": "profitability_w",
    "Cash holdings": "cash_w",
    "Market-to-book": "market_to_book_w"
}

H2_MODEL_SPECS = []

for k in ADJUSTMENT_HORIZONS:
    H2_MODEL_SPECS.append({
        "name": f"H2 M{k} {k}-year adjustment",
        "dep": f"delta_leverage_deviation_{k}",
        "xvars": ["leverage_deficit_w", "firm_acq_intensity_w", f"target_leverage_change_{k}"],
        "controls": H2_CONTROLS,
        "common_sample_vars": H2_COMMON_BASE_VARS + [
            f"delta_leverage_deviation_{k}", f"target_leverage_change_{k}"
        ],
        "filter_condition": lambda d: d["firm_acq_intensity_w"] > 0,
        "year_fe": True,
        "industry_fe": True,
        "display_vars": {
            "Leverage deviation": "leverage_deficit_w",
            "Acquisition intensity": "firm_acq_intensity_w",
            "Target leverage change": f"target_leverage_change_{k}",
            "Size": "size_w",
            "Profitability": "profitability_w",
            "Cash holdings": "cash_w",
            "Market-to-book": "market_to_book_w"
        }
    })

# Stricter HKW-style proxy restricted to large-acquisition firm-years
H2_LARGE_MODEL_SPECS = []

for k in ADJUSTMENT_HORIZONS:
    H2_LARGE_MODEL_SPECS.append({
        "name": f"H2 large M{k} {k}-year adjustment",
        "dep": f"delta_leverage_deviation_{k}",
        "xvars": ["leverage_deficit_w", "firm_acq_intensity_w", f"target_leverage_change_{k}"],
        "controls": H2_CONTROLS,
        "common_sample_vars": H2_COMMON_BASE_VARS + [
            f"delta_leverage_deviation_{k}", f"target_leverage_change_{k}"
        ],
        "filter_condition": lambda d: d["large_acq_dummy"] == 1,
        "year_fe": True,
        "industry_fe": True,
        "display_vars": {
            "Leverage deviation": "leverage_deficit_w",
            "Acquisition intensity": "firm_acq_intensity_w",
            "Target leverage change": f"target_leverage_change_{k}",
            "Size": "size_w",
            "Profitability": "profitability_w",
            "Cash holdings": "cash_w",
            "Market-to-book": "market_to_book_w"
        }
    })


# H3: two-step acquisition decision model
H3_CONTROLS = [
    "size_w", "profitability_w", "tangibility_w", "cash_w",
    "market_to_book_w", "rd_intensity_w", "xrd_missing"
]

H3_COMMON_SAMPLE_VARS = [
    "future_firm_acq_intensity_1", "future_acq_dummy_1", "future_large_acq_dummy_1",
    "leverage_deficit_w", "overleveraged", "underleveraged",
    "gvkey", "fyear", "sic3"
] + H3_CONTROLS

H3_TABLE_ROWS = {
    "Leverage deficit": "leverage_deficit_w",
    "Overleveraged dummy": "overleveraged",
    "Underleveraged dummy": "underleveraged",
    "Size": "size_w",
    "Profitability": "profitability_w",
    "Tangibility": "tangibility_w",
    "Cash holdings": "cash_w",
    "Market-to-book": "market_to_book_w",
    "R&D intensity": "rd_intensity_w",
    "R&D missing indicator": "xrd_missing"
}

H3_MODEL_SPECS = [
    {
        "name": "H3 M1 Future acq intensity",
        "dep": "future_firm_acq_intensity_1",
        "xvars": ["leverage_deficit_w"],
        "controls": [],
        "common_sample_vars": H3_COMMON_SAMPLE_VARS,
        "year_fe": True,
        "industry_fe": False
    },
    {
        "name": "H3 M2 Future acq intensity full",
        "dep": "future_firm_acq_intensity_1",
        "xvars": ["leverage_deficit_w"],
        "controls": H3_CONTROLS,
        "common_sample_vars": H3_COMMON_SAMPLE_VARS,
        "year_fe": True,
        "industry_fe": True
    },
    {
        "name": "H3 M3 Over/under leverage",
        "dep": "future_firm_acq_intensity_1",
        "xvars": ["overleveraged", "underleveraged"],
        "controls": H3_CONTROLS,
        "common_sample_vars": H3_COMMON_SAMPLE_VARS,
        "year_fe": True,
        "industry_fe": True
    },
    {
        "name": "H3 M4 Future acquirer dummy",
        "dep": "future_acq_dummy_1",
        "xvars": ["leverage_deficit_w"],
        "controls": H3_CONTROLS,
        "common_sample_vars": H3_COMMON_SAMPLE_VARS,
        "year_fe": True,
        "industry_fe": True
    },
    {
        "name": "H3 M5 Future large acq dummy",
        "dep": "future_large_acq_dummy_1",
        "xvars": ["leverage_deficit_w"],
        "controls": H3_CONTROLS,
        "common_sample_vars": H3_COMMON_SAMPLE_VARS,
        "year_fe": True,
        "industry_fe": True
    }
]

# Regression helpers

def star(p):
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""


def format_coef(model, var):
    if var not in model.params.index:
        return ""

    coef = model.params[var]
    tval = model.tvalues[var]
    pval = model.pvalues[var]

    return f"{coef:.4f}{star(pval)} ({tval:.2f})"


def run_one_model(spec, data):
    temp = data.copy()
    industry_var = spec.get("industry_var", "sic3")

    # force common sample if requested
    common_sample_vars = spec.get("common_sample_vars", None)
    if common_sample_vars is not None:
        temp = temp.dropna(subset=common_sample_vars).copy()

    # optional sample restriction
    if "filter_condition" in spec and spec["filter_condition"] is not None:
        temp = temp.loc[spec["filter_condition"](temp)].copy()

    rhs_vars = spec["xvars"].copy()
    rhs_vars = rhs_vars + spec.get("controls", [])

    formula_parts = rhs_vars.copy()

    if spec["year_fe"]:
        formula_parts.append("C(fyear)")

    if spec["industry_fe"]:
        formula_parts.append(f"C({industry_var})")

    formula = spec["dep"] + " ~ " + " + ".join(formula_parts)

    needed = [spec["dep"], "gvkey", "fyear"] + rhs_vars

    if spec["industry_fe"]:
        needed.append(industry_var)

    needed = list(dict.fromkeys(needed))
    temp = temp.dropna(subset=needed).copy()

    if len(temp) == 0:
        raise ValueError(f"No observations left for {spec['name']}")

    if temp["gvkey"].nunique() > 1:
        model = smf.ols(formula, data=temp).fit(
            cov_type="cluster",
            cov_kwds={"groups": temp["gvkey"]}
        )
    else:
        model = smf.ols(formula, data=temp).fit(cov_type="HC1")

    return {
        "name": spec["name"],
        "model": model,
        "sample": temp,
        "formula": formula,
        "spec": spec
    }


def run_model_group(model_specs, data):
    results = []

    for spec in model_specs:
        result = run_one_model(spec, data)
        results.append(result)
        print(result["name"], "done. N =", int(result["model"].nobs))

    return results


def create_output_table(results, table_rows, save_path):
    reg_table = pd.DataFrame(index=list(table_rows.keys()))

    for result in results:
        model = result["model"]
        name = result["name"]
        spec = result["spec"]

        display_vars = spec.get("display_vars", table_rows)

        col = []
        for row_name in table_rows.keys():
            var = display_vars.get(row_name, table_rows[row_name])
            col.append(format_coef(model, var))

        reg_table[name] = col

    info_rows = pd.DataFrame(index=[
        "Observations",
        "Unique firms",
        "R-squared",
        "Dependent variable",
        "Year fixed effects",
        "Industry fixed effects",
        "Clustered standard errors"
    ])

    for result in results:
        model = result["model"]
        sample = result["sample"]
        spec = result["spec"]
        name = result["name"]

        info_rows[name] = [
            int(model.nobs),
            sample["gvkey"].nunique(),
            round(model.rsquared, 3),
            spec["dep"],
            "Yes" if spec["year_fe"] else "No",
            "Yes" if spec["industry_fe"] else "No",
            "Firm level"
        ]

    final_table = pd.concat([reg_table, info_rows])
    display(final_table)

    final_table.to_csv(save_path)
    print("table saved to:", save_path)

    return final_table


# Estimate models

print("running H1 peer benchmark models...")
h1_results = run_model_group(H1_MODEL_SPECS, DATA)

print("running H2 acquisition adjustment models...")
h2_results = run_model_group(H2_MODEL_SPECS, DATA)

print("running H2 large-acquisition adjustment models...")
h2_large_results = run_model_group(H2_LARGE_MODEL_SPECS, DATA)

print("running H3 acquisition decision models...")
h3_results = run_model_group(H3_MODEL_SPECS, DATA)

print("All regressions finished.")


# Export regression tables

h1_table = create_output_table(
    h1_results,
    H1_TABLE_ROWS,
    f"{OUTPUT_FOLDER}/h1_peer_benchmark_table.csv"
)

h2_table = create_output_table(
    h2_results,
    H2_TABLE_ROWS,
    f"{OUTPUT_FOLDER}/h2_acquisition_adjustment_table.csv"
)

h2_large_table = create_output_table(
    h2_large_results,
    H2_TABLE_ROWS,
    f"{OUTPUT_FOLDER}/h2_large_acquisition_adjustment_table.csv"
)

h3_table = create_output_table(
    h3_results,
    H3_TABLE_ROWS,
    f"{OUTPUT_FOLDER}/h3_acquisition_decision_table.csv"
)

# Export final data snapshot

export_cols = [
    "gvkey", "conm", "tic", "fyear", "sic", "sic2", "sic3",
    "market_leverage_w", "peer_leverage_w", "book_leverage_w",
    "target_leverage", "leverage_deficit_w", "overleveraged", "underleveraged",
    "firm_acq_intensity_w", "industry_acq_intensity_w", "acq_dummy", "large_acq_dummy",
    "future_firm_acq_intensity_1",
    "future_acq_dummy_1", "future_large_acq_dummy_1"
]

export_cols = [col for col in export_cols if col in df_clean.columns]

df_clean[export_cols].to_csv(
    f"{OUTPUT_FOLDER}/final_model_variables_snapshot.csv",
    index=False
)

print("final model variables snapshot saved.")
