import sys
import difflib
from collections import Counter
import pandas as pd

FILES = {"2025": "2025/results.csv", "2026": "2026/survey_results_public.csv"}


def columns(year, text=""):
    cols = pd.read_csv(FILES[year], nrows=0).columns
    hits = [c for c in cols if text.lower() in c.lower()]
    print(f"\n== {year}: {len(hits)} of {len(cols)} columns match '{text}'")
    print(", ".join(hits))


def peek(year, col, top=40):
    try:
        s = pd.read_csv(FILES[year], usecols=[col], dtype=str,
                        keep_default_na=False)[col]
    except ValueError:
        allcols = pd.read_csv(FILES[year], nrows=0).columns
        close = difflib.get_close_matches(col, allcols, n=5)
        print(f"\n== {year}: column '{col}' not found. Close names: {close}")
        return
    missing = s.isin(["", "NA"])
    print(f"\n== {year} | {col} | rows={len(s):,} | "
          f"missing={missing.sum():,} ({missing.mean():.1%})")
    present = s[~missing]
    items = Counter()
    for cell in present:
        for x in cell.split(";"):
            items[x.strip()] += 1
    print(f"cells with more than one item: {present.str.contains(';').mean():.1%}")
    for label, n in items.most_common(top):
        print(f"{n:>7,}  {label}")
    if len(items) > top:
        print(f"... and {len(items) - top} more labels")


if __name__ == "__main__":
    if sys.argv[1] == "--columns":
        columns(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
    else:
        for y in (sys.argv[2:] or FILES):
            peek(y, sys.argv[1])