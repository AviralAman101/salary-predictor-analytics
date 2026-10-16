import sys
import pandas as pd

FILES = {"2025": "2025/results.csv", "2026": "2026/survey_results_public.csv"}


def num(year, col):
    s = pd.read_csv(FILES[year], usecols=[col], dtype=str,
                    keep_default_na=False)[col].str.strip()
    blank = s.isin(["", "NA"])
    x = pd.to_numeric(s.where(~blank), errors="coerce")
    bad = s[~blank & x.isna()]
    print(f"\n== {year} | {col} | rows={len(s):,} | blank={blank.sum():,} "
          f"| numeric={x.notna().sum():,} | text values={len(bad):,}")
    if len(bad):
        print("most common text values:")
        print(bad.value_counts().head(10).to_string())
    print("percentiles of the numeric values:")
    print(x.dropna().quantile([0, .01, .05, .25, .5, .75, .95, .99, 1])
          .round(1).to_string())
    print(f"zero or negative: {(x <= 0).sum():,}")


if __name__ == "__main__":
    for y in (sys.argv[2:] or FILES):
        num(y, sys.argv[1])