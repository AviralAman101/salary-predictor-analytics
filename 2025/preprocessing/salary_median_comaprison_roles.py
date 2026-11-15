from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

csv_path = Path(__file__).resolve().parents[1] / "stage1_2025.csv"
stage1_2025 = pd.read_csv(csv_path)

chart_data = stage1_2025[
    (stage1_2025["is_developer"] == 1)
    & (stage1_2025["is_employed"] == 1)
    & (stage1_2025["salary_usd"].between(1_000, 500_000))
    ]

role_summary = (
    chart_data.groupby("role_group")
    .agg(
        salary_n=("salary_usd", "count"),
        median_salary=("salary_usd", "median"),
        experience_n=("work_exp", "count"),
        median_work_exp=("work_exp", "median"),
    )
    .sort_values("median_salary")
)

# Print the summary table, including the experience sample size
print(role_summary.round(1).to_string())

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(role_summary.index, role_summary["median_salary"])
ax.invert_yaxis()

ax.set_title("Median Annual Salary by Role — Stack Overflow 2025")
ax.set_xlabel("Median annual salary (USD)")
ax.xaxis.set_major_formatter(mtick.StrMethodFormatter("${x:,.0f}"))
ax.xaxis.grid(True, color="#DDDDDD")
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, role_summary["median_salary"].max() * 1.35)

for bar, (_, row) in zip(bars, role_summary.iterrows()):
    experience = (
        f"{row['median_work_exp']:.1f} yrs exp"
        if pd.notna(row["median_work_exp"])
        else "experience unavailable"
    )
    ax.text(
        bar.get_width() + role_summary["median_salary"].max() * 0.015,
        bar.get_y() + bar.get_height() / 2,
        f"Med. Sal: ${row['median_salary']:,.0f} ·| Med. EXP:{experience} |· n={int(row['salary_n']):,}",
        va="center",
        fontsize=8,
        )

fig.tight_layout()
plt.show()