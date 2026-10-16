import pandas as pd

CFG = [
    ("2025", "2025/results.csv", "MainBranch", "I am a developer by profession"),
    ("2026", "2026/survey_results_public.csv", "CodeMain",
     "I create code as a primary part of my job"),
]
for year, path, dev_col, dev_val in CFG:
    d = pd.read_csv(path, usecols=["Country", "Employment", dev_col],
                    dtype=str, keep_default_na=False)
    is_dev = d[dev_col] == dev_val
    groups = {
        "India developers": is_dev & (d["Country"] == "India"),
        "Other developers": is_dev & (d["Country"] != "India") & (d["Country"] != ""),
    }
    for name, mask in groups.items():
        print(f"\n{year} | {name} | n={mask.sum():,}")
        shares = d[mask]["Employment"].replace("", "(blank)").value_counts(normalize=True)
        print((shares * 100).round(1).to_string())