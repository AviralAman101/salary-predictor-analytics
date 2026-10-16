import sys
from collections import Counter
import pandas as pd
from common import load

year, gate, target = sys.argv[1], sys.argv[2], sys.argv[3]
f = load(year, [gate, target])
blank = f[target] == ""


def labels(cell):
    return {i.strip() for i in cell.split(";")} if cell else set()


sets = f[gate].map(labels)
items = Counter(x for s in sets for x in s)
print(f"{year} | final rows {len(f):,} | {target} blank {blank.mean():.1%}")
print(f"{gate} itself blank: {(f[gate] == '').sum():,}")
rows = []
for label, n in items.most_common(30):
    sel = sets.map(lambda s: label in s)
    rows.append((label[:60], n,
                 round(blank[sel].mean() * 100, 1),
                 round(blank[~sel].mean() * 100, 1) if (~sel).any() else None))
print(pd.DataFrame(rows, columns=["label", "selected",
                                  "blank_%_if_selected", "blank_%_if_not"])
      .to_string(index=False))