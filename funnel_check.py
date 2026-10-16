import pandas as pd

CFG = {
    "2025": dict(path="2025/results.csv", dev="MainBranch",
                 dev_val="I am a developer by profession"),
    "2026": dict(path="2026/survey_results_public.csv", dev="CodeMain",
                 dev_val="I create code as a primary part of my job"),
}
LOW, HIGH = 1_000, 500_000


def funnel(year, c):
    cols = ["Employment", c["dev"], "Country", "ConvertedCompYearly",
            "WorkExp", "YearsCode"]
    d = pd.read_csv(c["path"], usecols=cols, dtype=str, keep_default_na=False)
    d = d.replace("NA", "")
    pay = pd.to_numeric(d["ConvertedCompYearly"].replace("", pd.NA),
                        errors="coerce")
    india = d["Country"] == "India"
    m = pd.Series(True, index=d.index)
    steps = {"all rows": m.copy()}
    m &= d[c["dev"]] == c["dev_val"]
    steps["developer"] = m.copy()
    m &= d["Employment"] == "Employed"
    steps["employed"] = m.copy()
    m &= d["Country"] != ""
    steps["has country"] = m.copy()
    m &= pay.notna()
    steps["has salary"] = m.copy()
    m &= pay.between(LOW, HIGH)
    steps[f"salary {LOW:,} to {HIGH:,}"] = m.copy()
    print(f"\n== {year}")
    print(f"{'step':<28}{'rows':>9}{'India':>8}")
    for k, v in steps.items():
        print(f"{k:<28}{v.sum():>9,}{(v & india).sum():>8,}")
    has_pay = steps["has salary"]
    print(f"salary below {LOW:,}: {(has_pay & (pay < LOW)).sum():,} "
          f"(India {(has_pay & (pay < LOW) & india).sum():,}); "
          f"above {HIGH:,}: {(has_pay & (pay > HIGH)).sum():,}")
    for col in ["WorkExp", "YearsCode"]:
        blank = d[col] == ""
        print(f"{col} blank in final rows: {(m & blank).sum():,} "
              f"(India {(m & blank & india).sum():,})")


for y, cfg in CFG.items():
    funnel(y, cfg)