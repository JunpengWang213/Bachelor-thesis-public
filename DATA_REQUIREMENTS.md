# Data requirements

The analysis uses annual Compustat North America company fundamentals for
fiscal years 2000–2024. The licensed source data is not included in this public
repository.

Save an authorised extract as `data.csv` in the repository root with the
following columns:

```text
costat, curcd, datafmt, indfmt, consol, gvkey, datadate, conm, tic, cusip,
cik, fic, naics, sic, fyear, at, ceq, che, dlc, dltt, optprcca, ppent,
ebit, oibdp, sale, xrd, aqc, csho, naicsh, prcc_c, prcc_f
```

The script restricts the extract to consolidated, industrial, standard-format
USD records for U.S.-incorporated firms and applies the remaining sample rules
in code.
