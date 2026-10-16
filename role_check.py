import pandas as pd

CFG = {
    "2025": dict(path="2025/results.csv", dev="MainBranch",
                 dev_val="I am a developer by profession", role="DevType"),
    "2026": dict(path="2026/survey_results_public.csv", dev="CodeMain",
                 dev_val="I create code as a primary part of my job",
                 role="Role"),
}
LOW, HIGH = 1_000, 500_000
GROUPS = {
    "Full-stack developer": ["Developer, full-stack"],
    "Back-end developer": ["Developer, back-end"],
    "Front-end developer": ["Developer, front-end"],
    "Mobile developer": ["Developer, mobile"],
    "Desktop/enterprise developer": ["Developer, desktop or enterprise applications"],
    "Game/graphics developer": ["Developer, game or graphics"],
    "Embedded/hardware": ["Developer, embedded applications or devices",
                          "Hardware engineer"],
    "QA/test": ["Developer, QA or test"],
    "Data and AI": ["Data engineer", "Data scientist", "AI/ML engineer",
                    "Applied scientist", "Developer, AI apps or physical AI"],
    "Analyst": ["Data or business analyst", "Financial analyst or engineer"],
    "DevOps/cloud/platform": ["DevOps engineer or professional",
                              "Cloud infrastructure engineer",
                              "Site reliability engineer", "Platform engineer"],
    "IT admin/database": ["System administrator",
                          "Database administrator or engineer"],
    "Architect": ["Architect, software or solutions"],
    "Security": ["Cybersecurity or InfoSec professional"],
    "Management": ["Engineering manager", "Senior executive (C-suite, VP, etc.)",
                   "Founder, technology or otherwise", "Founder or entrepreneur",
                   "Project manager", "Product manager"],
    "Other roles": ["Other (please specify):", "Other", "Academic researcher",
                    "Educator or instructor", "Student", "Retired",
                    "Support engineer or analyst",
                    "UX, Research Ops or UI design professional",
                    "UX, research, or UI design professional",
                    "Solutions engineer, implementation engineer, or forward-deployed engineer",
                    "Software engineer",
                    "Developer advocate or developer relations professional",
                    "Technical writer or documentation professional"],
}
LABEL_TO_GROUP = {lab: g for g, labs in GROUPS.items() for lab in labs}

for year, c in CFG.items():
    cols = ["Employment", c["dev"], "Country", "ConvertedCompYearly", c["role"]]
    d = pd.read_csv(c["path"], usecols=cols, dtype=str,
                    keep_default_na=False).replace("NA", "")
    pay = pd.to_numeric(d["ConvertedCompYearly"].replace("", pd.NA),
                        errors="coerce")
    keep = ((d[c["dev"]] == c["dev_val"]) & (d["Employment"] == "Employed")
            & (d["Country"] != "") & pay.between(LOW, HIGH))
    f = d[keep]
    role = f[c["role"]]
    group = role.map(LABEL_TO_GROUP).where(role != "", "Not stated")
    bad = role[group.isna()].value_counts()
    group = group.fillna("UNMAPPED")
    is_india = f["Country"] == "India"
    out = pd.DataFrame({"rows": group.value_counts(),
                        "India": group[is_india].value_counts()}).fillna(0).astype(int)
    out["share_%"] = (out["rows"] / len(f) * 100).round(1)
    print(f"\n== {year} | final rows {len(f):,}")
    print(out.to_string())
    if len(bad):
        print("\nUNMAPPED labels:")
        print(bad.to_string())