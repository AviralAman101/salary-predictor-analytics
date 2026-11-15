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

countries = [
    "United States of America",
    "Germany",
    "United Kingdom",
    "India",
    "France",
]

data = df[
    (df["is_developer"] == 1)
    & (df["is_employed"] == 1)
    & df["salary_usd"].between(1_000, 500_000)
    & df["country"].isin(countries)
    & (df["work_setting"] != "Not stated")
    ].copy()

work_order = [
    "Remote",
    "Your choice (very flexible, you can come in when you want or just as needed)",
    "Hybrid (some in-person, leans heavy to flexibility)",
    "Hybrid (some remote, leans heavy to in-person)",
    "In-person",
]

summary = (
    data.groupby(["work_setting", "country"])["salary_usd"]
    .agg(n="count", mean="mean", median="median")
    .reset_index()
)

mean_table = (
    summary.pivot(index="work_setting", columns="country", values="mean")
    .reindex(index=work_order, columns=countries)
)
count_table = (
    summary.pivot(index="work_setting", columns="country", values="n")
    .reindex(index=work_order, columns=countries)
)

print("Mean annual salary (nominal USD)")
print(mean_table.round(0).to_string())
print("\nRespondent count")
print(count_table.fillna(0).astype(int).to_string())

short_labels = {
    "Your choice (very flexible, you can come in when you want or just as needed)": "Flexible choice",
    "Hybrid (some in-person, leans heavy to flexibility)": "Hybrid — leans flexible",
    "Hybrid (some remote, leans heavy to in-person)": "Hybrid — leans in-person",
}
plot_table = mean_table.rename(index=short_labels).rename(
    columns={
        "United States of America": "United States",
        "United Kingdom": "United Kingdom",
    }
)

ax = plot_table.plot(
    kind="bar",
    figsize=(12, 6),
    width=0.82,
    color=["#2878B5", "#E07A3F", "#4A9B75", "#8B6AB1", "#C5A43A"],
)

ax.set_title(
    "Mean Developer Salary by Work Setting and Country — 2025",
    loc="left",
    fontsize=14,
    fontweight="bold",
)
ax.set_xlabel("Work setting")
ax.set_ylabel("Mean annual salary (USD)")
ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
ax.grid(axis="y", color="#DDDDDD")
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(axis="x", rotation=15)
ax.legend(title="Country", frameon=False, ncol=3)
plt.tight_layout()
plt.show()