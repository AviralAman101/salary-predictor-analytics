import pandas as pd

CFG = {
    "2025": dict(path="2025/results.csv", dev="MainBranch",
                 dev_val="I am a developer by profession"),
    "2026": dict(path="2026/survey_results_public.csv", dev="CodeMain",
                 dev_val="I create code as a primary part of my job"),
}
LOW, HIGH = 1_000, 500_000
MAX_EXP = {"Under 18 years old": 2, "18-24 years old": 8, "25-34 years old": 18,
           "35-44 years old": 28, "45-54 years old": 38, "55-64 years old": 48,
           "65 years or older": 55, "Prefer not to say": 50}
ORDER = list(MAX_EXP)

for year, c in CFG.items():
    cols = ["Employment", c["dev"], "Country", "ConvertedCompYearly",
            "WorkExp", "Age"]
    d = pd.read_csv(c["path"], usecols=cols, dtype=str,
                    keep_default_na=False).replace("NA", "")
    pay = pd.to_numeric(d["ConvertedCompYearly"].replace("", pd.NA),
                        errors="coerce")
    keep = ((d[c["dev"]] == c["dev_val"]) & (d["Employment"] == "Employed")
            & (d["Country"] != "") & pay.between(LOW, HIGH))
    f = d[keep].copy()
    f["exp"] = pd.to_numeric(f["WorkExp"].replace("", pd.NA), errors="coerce")
    too_high = f["exp"] > f["Age"].map(MAX_EXP)
    is_india = f["Country"] == "India"
    print(f"\n== {year} | final rows {len(f):,}")
    print(f"Under 18 in final rows: {(f['Age'] == 'Under 18 years old').sum():,}")
    print(f"WorkExp above age-based limit: {too_high.sum():,} "
          f"(India {(too_high & is_india).sum():,})")
    for name, sub in [("all countries", f), ("India", f[is_india])]:
        share = (sub["Age"].value_counts(normalize=True)
                 .reindex(ORDER).fillna(0) * 100)
        print(f"\nage mix, {name} (n={len(sub):,}), %")
        print(share.round(1).to_string())