"""Create the self-contained analysis notebook in this folder."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(True)}

cells = [
md(r'''# Stack Overflow Developer Salary Analysis

This self-contained notebook starts with the original Stack Overflow survey downloads, harmonizes both years, and carries out descriptive, predictive, and prescriptive salary analysis. In Google Colab, upload the two raw CSVs into the same working directory as the notebook; no raw-data subfolder is needed. The accepted filename pairs are `results.csv` + `survey_results_public.csv`, or `2025_raw.csv` + `2026_raw.csv`.

## Run order

1. Install the listed Python packages in the notebook environment.
2. In Google Colab, upload the raw 2025 and 2026 CSVs into the notebook's working directory, then run top to bottom. The notebook also supports the repository's local `unharmonized_raw_data/` folder. Preprocessing runs **2026 first**, then 2025.
3. The preprocessing cells write harmonized `2026.csv` and `2025.csv` at the `Workbook` root, plus `harmonized_schema.csv`. Later cells create tables and charts under `outputs/`.
4. The predictive section tunes the random forest and fits linear regression using 2025 data, then compares both on 2026 data.

The notebook writes harmonized data and its schema into the current working directory, and creates `outputs/` automatically for tables and charts.'''),
md(r'''## 1. Packages and paths

This section imports the data, plotting, and machine learning libraries and prepares an output directory. Install any missing packages in your environment before continuing.'''),
code(r'''from pathlib import Path
import json
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
ROOT = Path.cwd()
filename_pairs = [("results.csv", "survey_results_public.csv"),
                  ("2025_raw.csv", "2026_raw.csv")]
RAW_DIR = ROOT
selected_pair = next((pair for pair in filename_pairs
                      if all((ROOT / name).exists() for name in pair)), None)
if selected_pair is None and (ROOT / "unharmonized_raw_data").exists():
    RAW_DIR = ROOT / "unharmonized_raw_data"
    selected_pair = next((pair for pair in filename_pairs
                          if all((RAW_DIR / name).exists() for name in pair)), None)
if selected_pair is None:
    candidates = [ROOT / "Workbook", ROOT / "workbook"]
    workbook = next((folder for folder in candidates
                     if (folder / "unharmonized_raw_data").exists()), None)
    if workbook is not None:
        ROOT = workbook
        RAW_DIR = ROOT / "unharmonized_raw_data"
        selected_pair = next((pair for pair in filename_pairs
                              if all((RAW_DIR / name).exists() for name in pair)), None)
if selected_pair is None:
    raise FileNotFoundError("Upload results.csv and survey_results_public.csv (or 2025_raw.csv and 2026_raw.csv) beside the notebook.")
RAW_2025 = RAW_DIR / selected_pair[0]
RAW_2026 = RAW_DIR / selected_pair[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
print("Workbook folder:", ROOT.resolve())
print("Outputs folder:", OUT.resolve())'''),
md(r'''## 2. Harmonization rules

The survey uses different names for several equivalent fields in each year. This section maps those source columns to a shared schema, normalizes role and skill labels, converts numeric fields, and defines the professional developer and employed indicators. The same rules are used for both years so later comparisons are consistent.'''),
code(r'''# Canonical field name -> source field in each raw survey.
SPECS = {
    2025: {
        "file": RAW_2025, "dev_value": "I am a developer by profession",
        "columns": {"response_id":"ResponseId", "dev":"MainBranch", "employment":"Employment",
            "country":"Country", "salary":"ConvertedCompYearly", "age":"Age", "work_exp":"WorkExp",
            "years_code":"YearsCode", "role":"DevType", "education":"EdLevel", "org_size":"OrgSize",
            "work_setting":"RemoteWork", "tool_count":"ToolCountWork", "lang":"LanguageHaveWorkedWith",
            "db":"DatabaseHaveWorkedWith", "web":"WebframeHaveWorkedWith",
            "ide":"DevEnvsHaveWorkedWith", "cloud":"PlatformHaveWorkedWith"}},
    2026: {
        "file": RAW_2026, "dev_value": "I create code as a primary part of my job",
        "columns": {"response_id":"ResponseId", "dev":"CodeMain", "employment":"Employment",
            "country":"Country", "salary":"ConvertedCompYearly", "age":"Age", "work_exp":"WorkExp",
            "years_code":"YearsCode", "role":"Role", "education":"Education", "org_size":"OrgSize",
            "work_setting":"WorkSetting", "tool_count":"ToolCount", "lang":"LanguagesHaveWorkedWith",
            "db":"DatabasesHaveWorkedWith", "web":"WebFrameworkHaveWorkedWith",
            "ide":"DevIDEHaveWorkedWith", "cloud":"CloudPlatHaveWorkedWith"}}
}

ROLE_GROUPS = {
    "Full-stack developer":["Developer, full-stack"],
    "Back-end developer":["Developer, back-end"],
    "Front-end developer":["Developer, front-end"],
    "Mobile developer":["Developer, mobile"],
    "Desktop/enterprise developer":["Developer, desktop or enterprise applications"],
    "Game/graphics developer":["Developer, game or graphics"],
    "Embedded/hardware":["Developer, embedded applications or devices", "Hardware engineer"],
    "QA/test":["Developer, QA or test"],
    "Data, analytics and AI":["Data engineer", "Data scientist", "AI/ML engineer", "Applied scientist",
        "Developer, AI apps or physical AI", "Data or business analyst", "Financial analyst or engineer"],
    "DevOps/cloud/platform":["DevOps engineer or professional", "Cloud infrastructure engineer",
        "Site reliability engineer", "Platform engineer"],
    "IT, security and admin":["System administrator", "Database administrator or engineer",
        "Cybersecurity or InfoSec professional"],
    "Architect":["Architect, software or solutions"],
    "Management":["Engineering manager", "Senior executive (C-suite, VP, etc.)", "Founder, technology or otherwise",
        "Founder or entrepreneur", "Project manager", "Product manager"],
    "Other roles":["Other (please specify):", "Other", "Academic researcher", "Educator or instructor", "Student",
        "Retired", "Support engineer or analyst", "UX, Research Ops or UI design professional",
        "UX, research, or UI design professional", "Solutions engineer, implementation engineer, or forward-deployed engineer",
        "Software engineer", "Developer advocate or developer relations professional",
        "Technical writer or documentation professional", "Marketing/Sales professional"]}
ROLE_MAP = {label.lower(): group for group, labels in ROLE_GROUPS.items() for label in labels}
SKILL_ALIASES = {"docker / docker compose":"docker", "digital ocean":"digitalocean", "maven (build tool)":"maven"}
AGE_MAX_EXP = {"Under 18 years old":2, "18-24 years old":8, "25-34 years old":18,
    "35-44 years old":28, "45-54 years old":38, "55-64 years old":48,
    "65 years or older":55, "Prefer not to say":50}
AGE_OK = [age for age in AGE_MAX_EXP if age != "Under 18 years old"]
SALARY_RANGE = (1_000, 500_000)
SKILL_FIELDS = ["lang_list", "db_list", "web_list", "ide_list", "cloud_list"]

print("Harmonized skill families:", SKILL_FIELDS)
print("Role groups:", len(ROLE_GROUPS))'''),
md(r'''## 3. Reusable preprocessing function

The function below reads only the fields needed for this analysis from a raw survey CSV. It standardizes the column names, groups detailed job titles into broader roles, harmonizes skill-list text, and preserves all respondent rows in the preprocessed file. Eligibility flags are added for later descriptive and predictive analysis; they do not delete raw respondents.'''),
code(r'''def clean_text(value):
    return str(value).replace("’", "'").strip()

def clean_skill_list(value):
    items = []
    for item in str(value).split(";"):
        key = clean_text(item).lower()
        key = SKILL_ALIASES.get(key, key)
        if key and not key.startswith("other") and key not in items:
            items.append(key)
    return ";".join(items)

def education_group(value):
    x = clean_text(value).lower()
    for prefix, group in [("primary", "Secondary or less"), ("secondary", "Secondary or less"),
        ("some college", "Some college"), ("associate", "Associate"), ("bachelor", "Bachelor's"),
        ("master", "Master's"), ("professional", "Doctorate/professional")]:
        if x.startswith(prefix):
            return group
    return "Other"

def preprocess_year(year):
    spec = SPECS[year]
    colmap = spec["columns"]
    raw = pd.read_csv(spec["file"], usecols=list(colmap.values()), dtype=str,
                      keep_default_na=False, low_memory=False)
    raw = raw.replace("NA", "")
    source = pd.DataFrame({key: raw[source_col].astype(str).str.strip()
                           for key, source_col in colmap.items()})
    d = pd.DataFrame(index=source.index)
    d["year"] = year
    d["id"] = str(year) + "_" + source["response_id"]
    d["is_developer"] = source["dev"].eq(spec["dev_value"]).astype("int8")
    d["is_employed"] = source["employment"].eq("Employed").astype("int8")
    d["country"] = source["country"].replace({"United Kingdom of Great Britain and Northern Ireland":"United Kingdom"})
    d["salary_usd"] = pd.to_numeric(source["salary"].replace("", np.nan), errors="coerce")
    d["age"] = source["age"].replace("", "Not stated")
    for col in ["work_exp", "years_code", "tool_count_raw"]:
        raw_col = {"work_exp":"work_exp", "years_code":"years_code", "tool_count_raw":"tool_count"}[col]
        d[col] = pd.to_numeric(source[raw_col].replace("", np.nan), errors="coerce")
    d["role_group"] = source["role"].str.lower().map(ROLE_MAP).fillna("Other roles")
    d["education"] = source["education"].map(education_group)
    d["org_size"] = source["org_size"].map(lambda x: clean_text(x) or "Not stated")
    d["work_setting"] = source["work_setting"].map(lambda x: clean_text(x) or "Not stated")
    for family in ["lang", "db", "web", "ide", "cloud"]:
        d[f"{family}_list"] = source[family].map(clean_skill_list)

    # A consistent eligibility flag used for salary analysis and modeling.
    age_limit = d["age"].map(AGE_MAX_EXP)
    eligible = (d["is_developer"].eq(1) & d["is_employed"].eq(1) & d["country"].ne("") &
        d["salary_usd"].between(*SALARY_RANGE) & d["age"].isin(AGE_OK) &
        ~(d["work_exp"] > age_limit) & d["years_code"].notna())
    d["in_model"] = eligible.astype("int8")
    d["has_salary"] = d["salary_usd"].notna().astype("int8")
    d["work_exp_blank"] = d["work_exp"].isna().astype("int8")
    d.loc[d["in_model"].eq(1) & d["work_exp"].isna(), "work_exp"] = 0
    d["tool_count"] = d["tool_count_raw"].where(d["tool_count_raw"] > 0).clip(upper=50)
    return d

print("Preprocessing function is ready. It keeps raw respondent rows and adds an analysis eligibility flag.")'''),
md(r'''## 4. Preprocess 2026 first

Run this cell first to harmonize the 2026 raw survey. The resulting shared-schema `2026.csv` is saved directly inside `Workbook`.'''),
code(r'''df_2026 = preprocess_year(2026)
path_2026 = ROOT / "2026.csv"
df_2026.to_csv(path_2026, index=False)
print(f"2026: {len(df_2026):,} respondents; {int(df_2026.in_model.sum()):,} eligible for salary modeling")
print("Saved:", path_2026.name)
display(df_2026.head())'''),
md(r'''## 5. Preprocess 2025 second

Apply the same normalization rules to the 2025 raw survey and save `2025.csv`. Both year files now share the same canonical columns.'''),
code(r'''df_2025 = preprocess_year(2025)
path_2025 = ROOT / "2025.csv"
df_2025.to_csv(path_2025, index=False)
print(f"2025: {len(df_2025):,} respondents; {int(df_2025.in_model.sum()):,} eligible for salary modeling")
print("Saved:", path_2025.name)
assert list(df_2025.columns) == list(df_2026.columns)
print("Schema check passed:", len(df_2025.columns), "shared columns")
display(df_2025.head())'''),
md(r'''## 6. Shared output schema

Both harmonized files use the same named columns and order. The schema reference records the meaning of each field and which original field it came from in each survey. In particular, `salary_usd` uses Stack Overflow’s `ConvertedCompYearly` value in both years; the respondent’s original currency/compensation pair is not mixed into that prediction target.'''),
code(r'''OUTPUT_DESCRIPTIONS = {
    "year":"Survey year.", "id":"Year-prefixed public response identifier.",
    "is_developer":"1 if the respondent selected the year-specific professional developer option.",
    "is_employed":"1 if employment status is Employed.", "country":"Country, with the UK label standardized.",
    "salary_usd":"Annual compensation converted to USD by the survey; sourced from ConvertedCompYearly.",
    "age":"Reported age band.", "work_exp":"Reported years of professional work experience.",
    "years_code":"Reported years coding.", "role_group":"Role mapped to a shared role group.",
    "education":"Education label grouped into shared categories.", "org_size":"Reported organization size.",
    "work_setting":"Reported work setting.", "tool_count_raw":"Raw survey count of tools/technologies used.",
    "lang_list":"Normalized semicolon-delimited programming language list.",
    "db_list":"Normalized semicolon-delimited database list.",
    "web_list":"Normalized semicolon-delimited web framework list.",
    "ide_list":"Normalized semicolon-delimited IDE list.",
    "cloud_list":"Normalized semicolon-delimited cloud/platform list.",
    "in_model":"1 if the row meets the shared salary-model eligibility criteria.",
    "has_salary":"1 if ConvertedCompYearly is present.",
    "work_exp_blank":"1 if reported professional work experience is missing.",
    "tool_count":"Tool count, positive values only and capped at 50."}
SOURCE_KEYS = {"id":"response_id", "is_developer":"dev", "is_employed":"employment",
               "salary_usd":"salary", "has_salary":"salary", "work_exp_blank":"work_exp",
               "role_group":"role", "education":"education",
               "tool_count_raw":"tool_count", "lang_list":"lang", "db_list":"db",
               "web_list":"web", "ide_list":"ide", "cloud_list":"cloud", "tool_count":"tool_count"}
schema_rows = []
for position, field in enumerate(df_2025.columns, start=1):
    key = SOURCE_KEYS.get(field, field)
    derived_rule = field == "in_model"
    source_2025 = "Derived eligibility rules" if derived_rule else SPECS[2025]["columns"].get(key, "Derived" if field == "year" else "")
    source_2026 = "Derived eligibility rules" if derived_rule else SPECS[2026]["columns"].get(key, "Derived" if field == "year" else "")
    schema_rows.append({
        "column_number":position, "harmonized_column":field,
        "source_column_2025":source_2025, "source_column_2026":source_2026,
        "description":OUTPUT_DESCRIPTIONS.get(field, "Normalized survey field."),
        "data_type":str(df_2025[field].dtype)})
harmonized_schema = pd.DataFrame(schema_rows)
harmonized_schema.to_csv(ROOT / "harmonized_schema.csv", index=False)
assert df_2025.columns.tolist() == df_2026.columns.tolist()
print("Saved shared schema with", len(harmonized_schema), "fields")
display(harmonized_schema)'''),
md(r'''### Reload point for continuing in a fresh kernel

This cell reads the completed harmonized files while preserving empty skill-list values as empty strings. Use it to resume analysis without rerunning the raw-data preprocessing cells.'''),
code(r'''df_2025 = pd.read_csv(ROOT / "2025.csv", keep_default_na=False, low_memory=False)
df_2026 = pd.read_csv(ROOT / "2026.csv", keep_default_na=False, low_memory=False)
NUMERIC_OUTPUTS = ["year", "is_developer", "is_employed", "salary_usd", "work_exp",
                   "years_code", "tool_count_raw", "in_model", "has_salary",
                   "work_exp_blank", "tool_count"]
for frame in [df_2025, df_2026]:
    for column in NUMERIC_OUTPUTS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
path_2025, path_2026 = ROOT / "2025.csv", ROOT / "2026.csv"
print("Loaded harmonized files with blank skill lists preserved:", len(df_2025), len(df_2026))'''),
md(r'''## 7. Preprocessing checks and descriptive overview

Before analysis, compare row counts, missing salary counts, role mix, and country representation. These summaries help identify differences in respondent composition across survey years.'''),
code(r'''summary = pd.DataFrame([
    {"year": year, "respondents": len(df), "developers": int(df.is_developer.sum()),
     "employed": int(df.is_employed.sum()), "with_salary": int(df.has_salary.sum()),
     "eligible_salary_rows": int(df.in_model.sum()), "countries": int(df.loc[df.country.ne(""), "country"].nunique())}
    for year, df in [(2025, df_2025), (2026, df_2026)]])
display(summary)
summary.to_csv(OUT / "respondent_summary.csv", index=False)

role_counts = pd.concat([df[df.in_model.eq(1)].assign(year=year).groupby(["year", "role_group"]).size().rename("respondents")
                         for year, df in [(2025, df_2025), (2026, df_2026)]])
display(role_counts.rename("eligible respondents").to_frame().head(20))'''),
md(r'''## 8. Descriptive analytics: salary by role

This section compares median salary by harmonized role group for eligible employed developers. The median is used because salary distributions often contain a small number of unusually high values.'''),
code(r'''role_salary = pd.concat([
    df.loc[df.in_model.eq(1)].assign(year=year).groupby(["year", "role_group"]).salary_usd
      .agg(respondents="count", median_salary="median", average_salary="mean").reset_index()
    for year, df in [(2025, df_2025), (2026, df_2026)]], ignore_index=True)
display(role_salary.sort_values(["year", "median_salary"], ascending=[True, False]).head(20))
role_salary.to_csv(OUT / "role_salary_summary.csv", index=False)

for year in [2025, 2026]:
    data = role_salary[role_salary.year.eq(year)].sort_values("median_salary")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(data.role_group, data.median_salary, color="#2878B5")
    ax.set_title(f"Median annual salary by role — {year}", loc="left", weight="bold")
    ax.set_xlabel("Median salary (USD)"); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / f"median_salary_by_role_{year}.png", dpi=160, bbox_inches="tight")
    plt.show()'''),
md(r'''## 9. Descriptive analytics: developer salaries by organization size and work setting

These views summarize average salary for employed developers by their reported organization size and work setting. The respondent count is shown with each bar so small cohorts can be interpreted cautiously.'''),
code(r'''def plot_group_salary(df, year, group, title, filename):
    data = (df.loc[df.in_model.eq(1)].groupby(group).salary_usd
            .agg(respondents="count", average_salary="mean", median_salary="median")
            .sort_values("average_salary"))
    display(data.sort_values("average_salary", ascending=False))
    data.to_csv(OUT / filename)
    fig, ax = plt.subplots(figsize=(10, max(5.5, .85 * len(data))))
    labels = [textwrap.fill(str(label), width=34, break_long_words=False) for label in data.index]
    bars = ax.barh(np.arange(len(data)), data.average_salary, color="#3987a8")
    ax.set_yticks(np.arange(len(data)), labels=labels)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_title(f"{title} — {year}", loc="left", weight="bold")
    ax.set_xlabel("Average annual salary (USD)"); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    ax.set_xlim(0, data.average_salary.max() * 1.25)
    for bar, n in zip(bars, data.respondents):
        ax.text(bar.get_width(), bar.get_y()+bar.get_height()/2, f"  n={n:,}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / f"{filename.replace('.csv','.png')}", dpi=160, bbox_inches="tight")
    plt.show()

for year, frame in [(2025, df_2025), (2026, df_2026)]:
    plot_group_salary(frame, year, "org_size", "Average salary by organization size", f"org_size_salary_{year}.csv")
    plot_group_salary(frame, year, "work_setting", "Average salary by work setting", f"work_setting_salary_{year}.csv")'''),
md(r'''## 10. Predictive analytics: 2025 training and 2026 testing

The model predicts annual salary in USD for the eligible professional developer cohort. The random forest settings below were selected by an earlier randomized search with three-fold cross-validation on 2025 data. They are now fixed so routine notebook runs train directly without repeating the search. The 2026 survey remains held for final evaluation.

We report MAE, RMSE, median absolute error, and R². Precision and classification accuracy are not appropriate for this continuous salary prediction task. Skill vocabularies are learned from 2025, then applied to 2026; unseen categories and tokens are safely ignored.'''),
code(r'''NUMERIC = ["work_exp", "years_code"]
CATEGORICAL = ["country", "age", "role_group", "education", "org_size", "work_setting"]
FEATURES = NUMERIC + CATEGORICAL + SKILL_FIELDS
train = df_2025.loc[df_2025.in_model.eq(1)].copy().reset_index(drop=True)
test = df_2026.loc[df_2026.in_model.eq(1)].copy().reset_index(drop=True)
X_train, y_train = train[FEATURES], train.salary_usd
X_test, y_test = test[FEATURES], test.salary_usd
print(f"Training rows (2025): {len(train):,}; test rows (2026): {len(test):,}")

def split_tokens(text):
    return [item.strip() for item in str(text).split(";") if item.strip()]

def make_pipeline(include_skills=True, params=None):
    transforms = [
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)]
    if include_skills:
        for col in SKILL_FIELDS:
            transforms.append((f"{col}_tokens", CountVectorizer(tokenizer=split_tokens,
                token_pattern=None, lowercase=False, binary=True, min_df=25), col))
    preprocessor = ColumnTransformer(transforms, remainder="drop")
    settings = {"n_estimators":120, "max_depth":None, "min_samples_split":20,
        "min_samples_leaf":3, "max_features":1.0, "max_samples":0.9,
        "random_state":42, "n_jobs":-1}
    if params:
        settings.update(params)
    forest = RandomForestRegressor(**settings)
    log_target = TransformedTargetRegressor(regressor=forest, func=np.log1p,
        inverse_func=np.expm1, check_inverse=False)
    return Pipeline([("features", preprocessor), ("model", log_target)])

RF_PARAMS = {"n_estimators":120, "max_depth":None, "min_samples_split":20,
    "min_samples_leaf":3, "max_features":1.0, "max_samples":0.9,
    "random_state":42, "n_jobs":-1}
best_model = make_pipeline(True, RF_PARAMS)
best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)
print("Random forest fit completed using previously selected fixed hyperparameters:", RF_PARAMS)'''),
code(r'''linear_model = Pipeline([
    ("features", make_pipeline(True).named_steps["features"]),
    ("model", TransformedTargetRegressor(regressor=LinearRegression(),
        func=np.log1p, inverse_func=np.expm1, check_inverse=False))])
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
print("Linear regression fitted with the same features and log-salary target transformation.")'''),
md(r'''## 11. Compare linear regression and random forest

Both models use the same 2025 training rows, 2026 test rows, feature columns, preprocessing, and log-salary target transformation. The random forest uses fixed settings selected during earlier 2025 cross-validation; this run does not repeat the search. Linear regression is fit directly on 2025. The metrics compare their 2026 predictions; lower error and higher R² indicate better results.'''),
code(r'''def regression_metrics(actual, predicted):
    return {"MAE_USD":mean_absolute_error(actual, predicted),
        "RMSE_USD":np.sqrt(mean_squared_error(actual, predicted)),
        "Median_AE_USD":median_absolute_error(actual, predicted),
        "R_squared":r2_score(actual, predicted)}

rf_metrics = regression_metrics(y_test, predictions)
linear_metrics = regression_metrics(y_test, linear_predictions)
comparison = pd.DataFrame([
    {"model":"Tuned random forest", **rf_metrics},
    {"model":"Linear regression", **linear_metrics}])
display(comparison.round(3))
comparison.to_csv(OUT / "predictive_model_metrics.csv", index=False)
pd.DataFrame({"actual_salary_usd":y_test, "predicted_random_forest_usd":predictions,
              "predicted_linear_regression_usd":linear_predictions}).to_csv(
    OUT / "predictions_2026.csv", index=False)
(OUT / "best_model_parameters.json").write_text(json.dumps(RF_PARAMS, indent=2, default=str))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
specs = [("MAE_USD", "MAE (USD) · lower is better", True),
         ("RMSE_USD", "RMSE (USD) · lower is better", True),
         ("Median_AE_USD", "Median absolute error (USD) · lower is better", True),
         ("R_squared", "R² · higher is better", False)]
colors = ["#167d9a", "#e28b3c"]
for ax, (metric, title, money) in zip(axes.flat, specs):
    bars = ax.bar(comparison.model, comparison[metric], color=colors)
    ax.set_title(title, weight="bold", fontsize=10)
    ax.tick_params(axis="x", rotation=10, labelsize=9)
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
    if metric == "R_squared": ax.axhline(0, color="#555", linewidth=.8)
    for bar, value in zip(bars, comparison[metric]):
        label = "$" + format(value, ",.0f") if money else format(value, ".3f")
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), label,
                ha="center", va="bottom", fontsize=9)
fig.suptitle("2026 holdout performance · trained on 2025", fontsize=14, weight="bold")
fig.tight_layout(rect=[0, 0, 1, .94]); fig.savefig(OUT / "model_metrics_comparison.png", dpi=170, bbox_inches="tight")
plt.show()

rng = np.random.default_rng(42)
sample_idx = rng.choice(len(y_test), min(3500, len(y_test)), replace=False)
actual_sample = y_test.iloc[sample_idx].to_numpy()
rf_sample = predictions[sample_idx]
linear_sample = linear_predictions[sample_idx]
fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
limit = max(actual_sample.max(), rf_sample.max(), linear_sample.max())
for ax, sample, name, color in zip(axes, [rf_sample, linear_sample],
        ["Tuned random forest", "Linear regression"], colors):
    ax.scatter(actual_sample, sample, s=10, alpha=.25, color=color, edgecolors="none")
    ax.plot([0, limit], [0, limit], "--", color="#555", label="Perfect prediction")
    ax.set(title=name, xlabel="Actual salary (USD)"); ax.grid(alpha=.2); ax.legend(frameon=False)
axes[0].set_ylabel("Predicted salary (USD)")
fig.suptitle("2026 holdout: predicted vs actual", weight="bold"); fig.tight_layout()
fig.savefig(OUT / "predicted_vs_actual.png", dpi=170, bbox_inches="tight"); plt.show()'''),
md(r'''## 12. Skill feature importance

Random forest feature importance indicates which encoded inputs the fitted model relied on most. It is an association within this model and dataset, not evidence that a skill causes higher pay.'''),
code(r'''preprocessor = best_model.named_steps["features"]
forest = best_model.named_steps["model"].regressor_
importance = pd.DataFrame({"feature":preprocessor.get_feature_names_out(),
                           "importance":forest.feature_importances_})
importance["feature"] = importance.feature.str.replace(r"^.*__", "", regex=True)
importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
importance.to_csv(OUT / "feature_importances.csv", index=False)
top = importance.head(20).sort_values("importance")
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top.feature, top.importance, color="#2878B5")
ax.set_title("Top model feature importances", loc="left", weight="bold")
ax.set_xlabel("Random forest feature importance"); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT / "top_feature_importances.png", dpi=170, bbox_inches="tight")
plt.show()
display(importance.head(20))'''),
md(r'''## 13. Random forest skill scenario analysis

For a small set of candidate skills, this section toggles one skill on for 2026 profiles who did not report it, then compares random forest predictions while holding the other answers fixed. This describes model sensitivity only; it does not estimate causal salary gains from learning a skill.'''),
code(r'''CANDIDATE_SKILLS = {
    "AWS":"cloud_list", "docker":"cloud_list", "kubernetes":"cloud_list",
    "terraform":"cloud_list", "java":"lang_list"}
scenario_rows = []
for skill, column in CANDIDATE_SKILLS.items():
    token = "amazon web services (aws)" if skill == "AWS" else skill
    is_present = test[column].map(lambda cell: token.lower() in {t.lower() for t in split_tokens(cell)})
    profiles = X_test.loc[~is_present].copy()
    if profiles.empty:
        continue
    changed = profiles.copy()
    changed[column] = changed[column].map(lambda cell: (cell + ";" if cell else "") + token)
    delta = best_model.predict(changed) - best_model.predict(profiles)
    scenario_rows.append({"skill":skill, "profiles_compared":len(delta),
        "mean_prediction_change_usd":float(np.mean(delta)),
        "median_prediction_change_usd":float(np.median(delta)),
        "share_with_increase_pct":float(np.mean(delta > 0) * 100)})
scenario = pd.DataFrame(scenario_rows).sort_values("mean_prediction_change_usd")
display(scenario.round(2))
scenario.to_csv(OUT / "candidate_skill_scenarios.csv", index=False)
if not scenario.empty:
    colors = ["#c56b55" if x < 0 else "#167d9a" for x in scenario.mean_prediction_change_usd]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(scenario.skill, scenario.mean_prediction_change_usd, color=colors)
    ax.axvline(0, color="#555", linewidth=.8)
    for bar, value in zip(bars, scenario.mean_prediction_change_usd):
        ax.text(value, bar.get_y()+bar.get_height()/2, f"  ${value:+,.0f}", va="center", fontsize=9)
    ax.set_title("Model sensitivity when adding one skill", loc="left", weight="bold")
    ax.set_xlabel("Mean change in predicted annual salary (USD)"); ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "candidate_skill_scenarios.png", dpi=170, bbox_inches="tight")
    plt.show()'''),
md(r'''## 14. Prescriptive analytics: rank an added skill for one fixed profile

This example selects one eligible 2026 respondent profile that does not already report any of the candidate skills. It keeps the person’s non-skill answers and all existing skills fixed, then adds one candidate skill at a time. Both trained models predict a salary for each scenario. The table ranks the candidates by their average predicted change across the two models and shows each model’s result separately.

This is a model-based scenario recommendation from observational survey data. It does not establish that learning a skill will cause a salary increase. The selected profile is an example; change the selection logic below if you want a different hypothetical profile.'''),
code(r'''# Select a reproducible example profile lacking all candidate skills.
def contains_skill(cell, token):
    return token.lower() in {item.lower() for item in split_tokens(cell)}

candidate_tokens = {"AWS":("cloud_list", "amazon web services (aws)"),
    "docker":("cloud_list", "docker"), "kubernetes":("cloud_list", "kubernetes"),
    "terraform":("cloud_list", "terraform"), "java":("lang_list", "java")}
eligible_for_example = pd.Series(True, index=test.index)
for label, (family, token) in candidate_tokens.items():
    eligible_for_example &= ~test[family].map(lambda cell: contains_skill(cell, token))
reported_skill_count = test[SKILL_FIELDS].apply(
    lambda column: column.map(lambda cell: len(split_tokens(cell)))).sum(axis=1)
eligible_for_example &= reported_skill_count.ge(8)
if not eligible_for_example.any():
    raise ValueError("No eligible 2026 profile has at least eight skills and lacks all candidates; adjust the selection rule.")
profile_index = test.loc[eligible_for_example].sample(n=1, random_state=42).index[0]
profile = X_test.loc[[profile_index]].copy()
profile_row = test.loc[profile_index]

profile_summary = pd.DataFrame([{"country":profile_row.country, "age":profile_row.age,
    "role_group":profile_row.role_group, "work_exp_years":profile_row.work_exp,
    "years_code":profile_row.years_code, "education":profile_row.education,
    "org_size":profile_row.org_size, "work_setting":profile_row.work_setting}])
print("Example profile (at least eight existing skills; all non-skill inputs held constant):")
display(profile_summary)
display(pd.DataFrame({"skill_family":SKILL_FIELDS,
    "existing_skills":[profile.iloc[0][field] for field in SKILL_FIELDS]}))

base_rf_salary = float(best_model.predict(profile)[0])
base_linear_salary = float(linear_model.predict(profile)[0])
prescription_rows = []
for label, (family, token) in candidate_tokens.items():
    changed = profile.copy()
    changed[family] = changed[family].map(lambda cell: (cell + ";" if cell else "") + token)
    rf_salary = float(best_model.predict(changed)[0])
    linear_salary = float(linear_model.predict(changed)[0])
    rf_change = rf_salary - base_rf_salary
    linear_change = linear_salary - base_linear_salary
    if rf_change > 0 and linear_change > 0: agreement = "Both predict an increase"
    elif rf_change < 0 and linear_change < 0: agreement = "Both predict a decrease"
    else: agreement = "Models disagree"
    prescription_rows.append({"candidate_skill":label,
        "base_random_forest_salary_usd":base_rf_salary,
        "random_forest_predicted_salary_usd":rf_salary,
        "random_forest_change_usd":rf_change,
        "base_linear_regression_salary_usd":base_linear_salary,
        "linear_regression_predicted_salary_usd":linear_salary,
        "linear_regression_change_usd":linear_change,
        "average_model_change_usd":(rf_change+linear_change)/2,
        "model_direction_agreement":agreement})
prescription = pd.DataFrame(prescription_rows).sort_values(
    "average_model_change_usd", ascending=False).reset_index(drop=True)
prescription.insert(0, "rank", np.arange(1, len(prescription)+1))
display(prescription.round(2))
prescription.to_csv(OUT / "prescriptive_skill_profile.csv", index=False)
if not prescription.empty:
    positions = np.arange(len(prescription)); width = .36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(positions-width/2, prescription.random_forest_change_usd, width,
           label="Tuned random forest", color="#167d9a")
    ax.bar(positions+width/2, prescription.linear_regression_change_usd, width,
           label="Linear regression", color="#e28b3c")
    ax.axhline(0, color="#555", linewidth=.8)
    ax.set_xticks(positions, prescription.candidate_skill)
    ax.set_ylabel("Change in predicted annual salary (USD)")
    ax.set_title("Added-skill scenarios for one fixed profile", loc="left", weight="bold")
    ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "prescriptive_skill_profile.png", dpi=170, bbox_inches="tight")
    plt.show()
best_scenario = prescription.iloc[0]
print("Highest-ranked scenario for this example profile:", best_scenario.candidate_skill,
      "(average predicted change $" + format(best_scenario.average_model_change_usd, ",.0f") + ").")'''),
md(r'''## 15. Outputs, interpretation, and limitations

Harmonized `2025.csv`, `2026.csv`, and `harmonized_schema.csv` are saved at the `Workbook` root. Analysis tables and charts are saved inside `Workbook/outputs/`. Both harmonized files preserve respondent rows and add the `in_model` eligibility flag. Model metrics and charts evaluate 2026 predictions from the fixed-parameter random forest and linear regression models.

Interpret the findings with these limits in mind:

- Survey respondents are self-selected and may not represent all software developers.
- Salary reporting, currencies, roles, and survey questions can differ across years even after field harmonization.
- A 2025-to-2026 test measures year-to-year generalization; it is not interchangeable with a random same-year split.
- Random forest feature importance and the skill recommendation are predictive associations, not causal estimates of the salary effect of gaining a skill.
- The salary range and cohort eligibility rules affect the reported results; document them in the final report.'''),
code(r'''print("Notebook run complete. Generated files:")
for path in sorted(OUT.iterdir()):
    print(" -", path.name)
print("Harmonized files:", path_2026.name, "and", path_2025.name)''')
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name":"Python 3", "language":"python", "name":"python3"},
        "language_info": {"name":"python", "version":"3.11+"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}
(HERE / "salary_analysis_workbook.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print("Created", HERE / "salary_analysis_workbook.ipynb")
