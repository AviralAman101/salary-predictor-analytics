"""Build the final 2025.csv and 2026.csv with identical columns.

Run after both prep scripts:  python3 common/finalize.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules  # noqa: E402

YEARS = (2025, 2026)
CATEGORICAL = ["age", "role_group", "education", "org_size", "work_setting",
               "country"]


def load_stage1(year):
    d = pd.read_csv(rules.ROOT / str(year) / f"stage1_{year}.csv", dtype=str,
                    keep_default_na=False)
    for c in ["year", "is_developer", "is_employed"]:
        d[c] = d[c].astype(int)
    for c in ["salary_usd", "work_exp", "years_code", "tool_count_raw"]:
        d[c] = rules.to_num(d[c])
    return d


def conditions(d):
    """The funnel, in order. A row is in the model if every step holds."""
    limit = d["age"].map(rules.AGE_MAX_EXP)
    return [
        ("developer", d["is_developer"] == 1),
        ("employed", d["is_employed"] == 1),
        ("has country", d["country"] != ""),
        ("has salary", d["salary_usd"].notna()),
        (f"salary {rules.SALARY_LOW:,} to {rules.SALARY_HIGH:,}",
         d["salary_usd"].between(rules.SALARY_LOW, rules.SALARY_HIGH)),
        ("age 18 or over", d["age"].isin(rules.AGES_OK)),
        ("WorkExp plausible for age", ~(d["work_exp"] > limit)),
        ("YearsCode present", d["years_code"].notna()),
    ]


def add_flags(d):
    m = np.ones(len(d), dtype=bool)
    for _, c in conditions(d):
        m &= c.to_numpy()
    d["in_model"] = m.astype(int)
    d["has_country"] = (d["country"] != "").astype(int)
    d["has_salary"] = d["salary_usd"].notna().astype(int)
    d["work_exp_blank"] = d["work_exp"].isna().astype(int)
    d.loc[(d["in_model"] == 1) & d["work_exp"].isna(), "work_exp"] = 0
    return d


def print_funnel(d, year):
    india = d["country"] == "India"
    m = pd.Series(True, index=d.index)
    print(f"\n== funnel {year}")
    print(f"{'step':<30}{'rows':>9}{'India':>8}")
    print(f"{'all rows':<30}{len(d):>9,}{india.sum():>8,}")
    for label, cond in conditions(d):
        m = m & cond
        print(f"{label:<30}{m.sum():>9,}{(m & india).sum():>8,}")


def row_sets(d, fam):
    return d[f"{fam}_list"].map(lambda s: set(s.split(";")) if s else set())


def shared_countries(frames):
    keep = None
    for d in frames.values():
        counts = d.loc[d["in_model"] == 1, "country"].value_counts()
        ok = set(counts[counts >= rules.MIN_COUNTRY_ROWS].index)
        keep = ok if keep is None else keep & ok
    if "India" not in keep:
        print("WARNING: India is below the country threshold in one year; "
              "keeping it anyway.")
        keep.add("India")
    return sorted(keep)


def shared_skills(frames, sets):
    out = {}
    for fam in rules.FAMILIES:
        counts = {}
        for y, d in frames.items():
            c = Counter()
            for st in sets[y][fam][d["in_model"] == 1]:
                c.update(st)
            counts[y] = c
        keys = [k for k in counts[YEARS[0]]
                if all(counts[y][k] >= rules.MIN_SKILL_USERS for y in YEARS)]
        out[fam] = {k: {y: counts[y][k] for y in YEARS} for k in sorted(keys)}
    return out


def tool_counts(frames):
    tcs = {}
    for y, d in frames.items():
        raw = d["tool_count_raw"]
        tcs[y] = raw.where(raw > 0).clip(upper=rules.TOOL_CAP)
    pooled = pd.concat([tcs[y][frames[y]["in_model"] == 1]
                        for y in YEARS]).median()
    return tcs, pooled


def build_final(d, tc, pooled, countries, skills, sets):
    out = pd.DataFrame({"id": d["id"], "year": d["year"]})
    for c in ["is_developer", "is_employed", "has_country", "has_salary",
              "in_model"]:
        out[c] = d[c]
    out["salary_usd"] = d["salary_usd"]
    out["log_salary"] = np.log(d["salary_usd"])
    out["country"] = d["country"].where(
        d["country"].isin(countries) | (d["country"] == ""), "Other")
    out["is_india"] = (d["country"] == "India").astype(int)
    out["work_exp"] = d["work_exp"]
    out["work_exp_blank"] = d["work_exp_blank"]
    out["years_code"] = d["years_code"]
    for c in ["age", "role_group", "education", "org_size", "work_setting"]:
        out[c] = d[c]
    out["tool_count"] = np.where(d["in_model"] == 1, tc.fillna(pooled), tc)
    out["toolcount_blank"] = tc.isna().astype(int)
    for fam in rules.FAMILIES:
        out[f"blank_{fam}"] = (d[f"{fam}_list"] == "").astype(int)
    cols = {}
    for fam in rules.FAMILIES:
        for k in skills[fam]:
            name = rules.col_name(fam, k)
            assert name not in cols, f"duplicate column name {name}"
            cols[name] = sets[fam].map(lambda st, k=k: int(k in st))
    return pd.concat([out, pd.DataFrame(cols)], axis=1)


def main():
    frames = {y: add_flags(load_stage1(y)) for y in YEARS}
    for y in YEARS:
        print_funnel(frames[y], y)

    sets = {y: {fam: row_sets(frames[y], fam) for fam in rules.FAMILIES}
            for y in YEARS}
    countries = shared_countries(frames)
    skills = shared_skills(frames, sets)
    tcs, pooled = tool_counts(frames)

    print(f"\nshared countries ({len(countries)}): {', '.join(countries)}")
    print(f"pooled median tool count (fills blanks): {pooled}")
    print("\nshared skills per family "
          f"(at least {rules.MIN_SKILL_USERS} users in each year):")
    for fam in rules.FAMILIES:
        print(f"  {fam}: {len(skills[fam])}")
    print("\ncloud and DevOps skills kept (users in 2025, 2026):")
    for k, n in sorted(skills["cloud"].items(),
                       key=lambda kv: -kv[1][YEARS[0]]):
        print(f"  {k:<28}{n[2025]:>7,}{n[2026]:>8,}")

    finals = {y: build_final(frames[y], tcs[y], pooled, countries, skills,
                             sets[y]) for y in YEARS}

    # ---- checks: the two files must line up ----
    a, b = finals[2025], finals[2026]
    assert list(a.columns) == list(b.columns), "column lists differ"
    print(f"\nboth files have the same {a.shape[1]} columns, in the same order")
    ok = True
    for c in CATEGORICAL:
        x = set(a.loc[a["in_model"] == 1, c])
        z = set(b.loc[b["in_model"] == 1, c])
        if x != z:
            ok = False
            print(f"label mismatch in {c}: only 2025 {sorted(x - z)} | "
                  f"only 2026 {sorted(z - x)}")
    if ok:
        print("category labels match in every shared column")

    for y, f in finals.items():
        path = rules.ROOT / str(y) / f"{y}.csv"
        f.to_csv(path, index=False)
        print(f"{y}: {len(f):,} rows ({int(f['in_model'].sum()):,} in the "
              f"model) written to {path}")

    lists = {"countries": countries,
             "skills": {fam: {k: {str(y): n for y, n in v.items()}
                              for k, v in skills[fam].items()}
                        for fam in rules.FAMILIES}}
    out = rules.ROOT / "common" / "shared_lists.json"
    out.write_text(json.dumps(lists, indent=2))
    print(f"shared lists saved to {out}")


if __name__ == "__main__":
    main()