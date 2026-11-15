from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

candidates = [
    Path("2025/stage1_2025.csv"),
    Path("../stage1_2025.csv"),
    Path("stage1_2025.csv"),
]
csv_path = next((p for p in candidates if p.exists()), None)
if csv_path is None:
    raise FileNotFoundError("Could not find stage1_2025.csv")

df = pd.read_csv(csv_path)

top_countries = [
    "United States of America",
    "Germany",
    "United Kingdom",
    "India",
    "France",
]

salary_cohort = df[
    (df["is_developer"] == 1)
    & (df["is_employed"] == 1)
    & df["salary_usd"].between(1_000, 500_000)
    & df["country"].isin(top_countries)
    ].copy()

size_map = {
    "Less than 20 employees": "Below 500",
    "20 to 99 employees": "Below 500",
    "100 to 499 employees": "Below 500",
    "500 to 999 employees": "500–4,999",
    "1,000 to 4,999 employees": "500–4,999",
    "5,000 to 9,999 employees": "5,000–9,999",
    "10,000 or more employees": "10,000+",
}

salary_cohort["org_cohort"] = salary_cohort["org_size"].map(size_map)
salary_cohort = salary_cohort.dropna(subset=["org_cohort"])

cohort_order = [
    "Below 500",
    "500–4,999",
    "5,000–9,999",
    "10,000+",
]

summary = (
    salary_cohort.groupby(["org_cohort", "country"])["salary_usd"]
    .agg(n="count", mean="mean", median="median")
    .reset_index()
)

mean_table = (
    summary.pivot(index="org_cohort", columns="country", values="mean")
    .reindex(index=cohort_order, columns=top_countries)
)
count_table = (
    summary.pivot(index="org_cohort", columns="country", values="n")
    .reindex(index=cohort_order, columns=top_countries)
)

print("Mean annual salary (nominal USD)")
print(mean_table.round(0).to_string())

print("\nRespondent count")
print(count_table.fillna(0).astype(int).to_string())

# Grouped bar chart generated at runtime from the analysis data
short_names = {
    "United States of America": "United States",
    "United Kingdom": "United Kingdom",
}
plot_table = mean_table.rename(columns=short_names)

ax = plot_table.plot(
    kind="bar",
    figsize=(12, 6),
    width=0.82,
    color=["#2878B5", "#E07A3F", "#4A9B75", "#8B6AB1", "#C5A43A"],
)

ax.set_title(
    "Mean Developer Salary by Organization Size and Country — 2025",
    loc="left",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Organization size")
ax.set_ylabel("Mean annual salary (USD)")
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
ax.grid(axis="y", color="#DDDDDD")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Country", frameon=False, ncol=3)
plt.tight_layout()
plt.show()