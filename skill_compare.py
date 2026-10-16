import sys
from collections import Counter
import pandas as pd
from common import load

MIN_N = 50


def counts(f, col):
    n = Counter()
    for cell in f[col]:
        if cell:
            for item in {x.strip() for x in cell.split(";")}:
                n[item] += 1
    return n


col25, col26 = sys.argv[1], sys.argv[2]
top = int(sys.argv[3]) if len(sys.argv) > 3 else 60
f25, f26 = load("2025", [col25]), load("2026", [col26])
c25, c26 = counts(f25, col25), counts(f26, col26)

df = pd.DataFrame([(k, c25[k], c26[k]) for k in set(c25) | set(c26)],
                  columns=["skill", "n2025", "n2026"])
df["pct2025"] = (df["n2025"] / len(f25) * 100).round(1)
df["pct2026"] = (df["n2026"] / len(f26) * 100).round(1)
df["both_ok"] = (df["n2025"] >= MIN_N) & (df["n2026"] >= MIN_N)
df = df.sort_values("n2025", ascending=False).reset_index(drop=True)

b25 = (f25[col25] == "").sum()
b26 = (f26[col26] == "").sum()
print(f"final rows: 2025 {len(f25):,} | 2026 {len(f26):,}")
print(f"blank list: 2025 {b25:,} ({b25 / len(f25):.1%}) | "
      f"2026 {b26:,} ({b26 / len(f26):.1%})")
print(f"labels: {len(df)} | usable in both years (>= {MIN_N} users each): "
      f"{int(df['both_ok'].sum())}")
print(df.head(top).to_string(index=False))
only = df[(df["n2025"] >= MIN_N) != (df["n2026"] >= MIN_N)]
print(f"\nlabels with >= {MIN_N} users in only one year:")
print(only[["skill", "n2025", "n2026"]].to_string(index=False))