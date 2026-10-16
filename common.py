import pandas as pd

CFG = {
    "2025": dict(path="2025/results.csv", dev="MainBranch",
                 dev_val="I am a developer by profession"),
    "2026": dict(path="2026/survey_results_public.csv", dev="CodeMain",
                 dev_val="I create code as a primary part of my job"),
}
LOW, HIGH = 1_000, 500_000


def load(year, extra_cols=()):
    """Final rows for one year: employed developers with a country and a
    salary between LOW and HIGH, plus the extra columns asked for."""
    c = CFG[year]
    cols = list(dict.fromkeys(
        ["Employment", c["dev"], "Country", "ConvertedCompYearly", *extra_cols]))
    d = pd.read_csv(c["path"], usecols=cols, dtype=str,
                    keep_default_na=False).replace("NA", "")
    pay = pd.to_numeric(d["ConvertedCompYearly"].replace("", pd.NA),
                        errors="coerce")
    keep = ((d[c["dev"]] == c["dev_val"]) & (d["Employment"] == "Employed")
            & (d["Country"] != "") & pay.between(LOW, HIGH))
    return d[keep].copy()