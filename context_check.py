import pandas as pd

CFG = {
    "2025": dict(path="2025/results.csv", dev="MainBranch",
                 dev_val="I am a developer by profession",
                 edu="EdLevel", work="RemoteWork"),
    "2026": dict(path="2026/survey_results_public.csv", dev="CodeMain",
                 dev_val="I create code as a primary part of my job",
                 edu="Education", work="WorkSetting"),
}
LOW, HIGH = 1_000, 500_000

IND = {
    "Software and internet": ["software development",
                              "internet, telecomm or information services",
                              "utilities, internet, or telecommunications"],
    "IT services and consulting": ["computer systems design and services",
                                   "it services, consulting, systems integration, or managed services"],
    "Financial services": ["fintech", "banking/financial services", "insurance"],
    "Healthcare and life sciences": ["healthcare", "life sciences or biotechnology"],
    "Manufacturing, energy, engineering": ["manufacturing", "energy",
                                           "engineering services or applied r&d", "mining or construction",
                                           "agriculture, forestry, fishing, and hunting"],
    "Public sector, education, research": ["government", "higher education",
                                           "education", "non-profit or social services", "scientific research"],
    "Retail, media, transport, other services": [
        "retail and consumer services", "retail, e-commerce, or consumer services",
        "media & advertising services", "media or advertising services",
        "transportation, or supply chain", "transportation or supply chain",
        "hospitality or travel", "broadcasting or publishing"],
    "Other": ["other:", "other, please specify"],
}
IND_MAP = {k: g for g, ks in IND.items() for k in ks}


def edu_group(x):
    x = x.replace("\u2019", "'").lower()
    if x == "":
        return "Not stated"
    for prefix, g in [("primary", "Secondary or less"), ("secondary", "Secondary or less"),
                      ("some college", "Some college"), ("associate", "Associate"),
                      ("bachelor", "Bachelor's"), ("master", "Master's"),
                      ("professional", "Doctorate/professional"), ("other", "Other")]:
        if x.startswith(prefix):
            return g
    return "UNMAPPED"


def ind_group(x):
    x = x.strip().lower()
    return "Not stated" if x == "" else IND_MAP.get(x, "UNMAPPED")


def show(title, group, is_india):
    out = pd.DataFrame({"rows": group.value_counts(),
                        "India": group[is_india].value_counts()}).fillna(0).astype(int)
    out["share_%"] = (out["rows"] / len(group) * 100).round(1)
    print(f"\n{title}")
    print(out.to_string())


for year, c in CFG.items():
    cols = ["Employment", c["dev"], "Country", "ConvertedCompYearly",
            c["edu"], "OrgSize", c["work"], "Industry"]
    d = pd.read_csv(c["path"], usecols=cols, dtype=str,
                    keep_default_na=False).replace("NA", "")
    pay = pd.to_numeric(d["ConvertedCompYearly"].replace("", pd.NA), errors="coerce")
    keep = ((d[c["dev"]] == c["dev_val"]) & (d["Employment"] == "Employed")
            & (d["Country"] != "") & pay.between(LOW, HIGH))
    f = d[keep]
    is_india = f["Country"] == "India"
    print(f"\n==== {year} | final rows {len(f):,}")
    print(f"blank OrgSize: {(f['OrgSize'] == '').sum():,}   "
          f"blank {c['work']}: {(f[c['work']] == '').sum():,}")
    show("Education groups", f[c["edu"]].map(edu_group), is_india)
    show("Industry groups", f["Industry"].map(ind_group), is_india)
    bad = f.loc[f["Industry"].map(ind_group) == "UNMAPPED", "Industry"].value_counts()
    if len(bad):
        print("\nUNMAPPED industry labels:")
        print(bad.to_string())