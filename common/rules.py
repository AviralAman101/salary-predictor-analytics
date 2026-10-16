"""Shared rules for both survey years.

The year scripts only translate column names. Everything that has to be
identical in 2025 and 2026 lives here.
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ---- agreed numbers -------------------------------------------------------
SALARY_LOW, SALARY_HIGH = 1_000, 500_000     # fixed USD bounds, both years
TOOL_CAP = 50                                # cap for the tool count
MIN_COUNTRY_ROWS = 100                       # per year, among modelling rows
MIN_SKILL_USERS = 150                        # per year, among modelling rows
FAMILIES = ["lang", "db", "web", "ide", "cloud"]

# Highest plausible WorkExp (years) for each age band (earliest start: age 16)
AGE_MAX_EXP = {
    "Under 18 years old": 2, "18-24 years old": 8, "25-34 years old": 18,
    "35-44 years old": 28, "45-54 years old": 38, "55-64 years old": 48,
    "65 years or older": 55, "Prefer not to say": 50,
}
AGES_OK = [a for a in AGE_MAX_EXP if a != "Under 18 years old"]

COUNTRY_FIX = {
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
}

# ---- role groups (14) -----------------------------------------------------
ROLE_GROUPS = {
    "Full-stack developer": ["Developer, full-stack"],
    "Back-end developer": ["Developer, back-end"],
    "Front-end developer": ["Developer, front-end"],
    "Mobile developer": ["Developer, mobile"],
    "Desktop/enterprise developer": ["Developer, desktop or enterprise applications"],
    "Game/graphics developer": ["Developer, game or graphics"],
    "Embedded/hardware": ["Developer, embedded applications or devices",
                          "Hardware engineer"],
    "QA/test": ["Developer, QA or test"],
    "Data, analytics and AI": ["Data engineer", "Data scientist", "AI/ML engineer",
                               "Applied scientist",
                               "Developer, AI apps or physical AI",
                               "Data or business analyst",
                               "Financial analyst or engineer"],
    "DevOps/cloud/platform": ["DevOps engineer or professional",
                              "Cloud infrastructure engineer",
                              "Site reliability engineer", "Platform engineer"],
    "IT, security and admin": ["System administrator",
                               "Database administrator or engineer",
                               "Cybersecurity or InfoSec professional"],
    "Architect": ["Architect, software or solutions"],
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
                    "Technical writer or documentation professional",
                    "Marketing/Sales professional"],
}
ROLE_MAP = {lab.lower(): g for g, labs in ROLE_GROUPS.items() for lab in labs}


# ---- small helpers --------------------------------------------------------
def norm(s):
    """Straight apostrophes, no surrounding spaces."""
    return s.replace("\u2019", "'").strip()


def education_group(x):
    x = norm(x).lower()
    for prefix, g in [("primary", "Secondary or less"),
                      ("secondary", "Secondary or less"),
                      ("some college", "Some college"),
                      ("associate", "Associate"), ("bachelor", "Bachelor's"),
                      ("master", "Master's"),
                      ("professional", "Doctorate/professional")]:
        if x.startswith(prefix):
            return g
    return "Other"            # includes "Other ..." and blanks


# Same tool, different label between the two years (keys are lower case)
SKILL_ALIASES = {
    "docker / docker compose": "docker",
    "digital ocean": "digitalocean",
    "maven (build tool)": "maven",
}


def skill_key(label):
    k = norm(label).lower()
    return SKILL_ALIASES.get(k, k)


def clean_list(cell):
    """'Python;SQL' -> 'python;sql' with aliases applied, 'Other' dropped."""
    items = []
    for x in cell.split(";"):
        k = skill_key(x)
        if k and not k.startswith("other") and k not in items:
            items.append(k)
    return ";".join(items)


def col_name(family, key):
    k = key.replace("#", "sharp").replace("+", "plus")
    return f"{family}_" + re.sub(r"[^0-9a-zA-Z]+", "_", k).strip("_")


def to_num(s):
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


# ---- stage 1: one year, canonical column names, all rows -------------------
def build_stage1(spec):
    cols = spec["cols"]
    raw = pd.read_csv(spec["path"], usecols=list(cols.values()), dtype=str,
                      keep_default_na=False).replace("NA", "")
    d = pd.DataFrame({k: raw[v].str.strip() for k, v in cols.items()})

    out = pd.DataFrame(index=d.index)
    out["year"] = spec["year"]
    out["id"] = str(spec["year"]) + "_" + d["response_id"]
    out["is_developer"] = (d["dev"] == spec["dev_value"]).astype(int)
    out["is_employed"] = (d["employment"] == "Employed").astype(int)
    out["country"] = d["country"].replace(COUNTRY_FIX)
    out["salary_usd"] = to_num(d["salary"])
    out["age"] = d["age"]
    out["work_exp"] = to_num(d["work_exp"])
    out["years_code"] = to_num(d["years_code"])

    role = d["role"]
    out["role_group"] = role.str.lower().map(ROLE_MAP).fillna("Other roles")
    unmapped = role[(role != "") & ~role.str.lower().isin(ROLE_MAP)]
    if len(unmapped):
        print("Role labels not in the map (counted as Other roles):")
        print(unmapped.value_counts().head(10).to_string())

    out["education"] = d["education"].map(education_group)
    out["org_size"] = d["org_size"].map(lambda x: norm(x) or "Not stated")
    out["work_setting"] = d["work_setting"].map(lambda x: norm(x) or "Not stated")
    out["tool_count_raw"] = to_num(d["tool_count"])
    for fam in FAMILIES:
        out[f"{fam}_list"] = d[fam].map(clean_list)

    out.to_csv(spec["stage1_path"], index=False)
    print(f"\n== stage 1 for {spec['year']} written to {spec['stage1_path']}")
    print(f"rows {len(out):,} | developers {out['is_developer'].sum():,} | "
          f"employed {out['is_employed'].sum():,} | "
          f"with salary {out['salary_usd'].notna().sum():,}")
    print("role groups:")
    print(out["role_group"].value_counts().to_string())
    print("blank skill lists (% of all rows):")
    for fam in FAMILIES:
        print(f"  {fam}: {(out[f'{fam}_list'] == '').mean():.1%}")
    return out