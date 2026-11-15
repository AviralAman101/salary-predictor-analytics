"""Train on Stack Overflow 2025 survey data and evaluate on 2026.

Run from the repository root with:
    python predictive_analysis/train_salary_model.py

Requires pandas, numpy, scikit-learn, and matplotlib.
Outputs are written to predictive_analysis/outputs/.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
SKILL_COLS = ["lang_list", "db_list", "web_list", "ide_list", "cloud_list"]
NUMERIC = ["work_exp", "years_code"]
CATEGORICAL = ["country", "age", "role_group", "education", "org_size", "work_setting"]
SKILLS_TO_COMPARE = ["amazon web services (aws)", "docker", "kubernetes", "terraform", "java"]
SEED = 42


def load_cohort(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = {"is_developer", "is_employed", "salary_usd", *NUMERIC, *CATEGORICAL, *SKILL_COLS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.loc[df.is_developer.eq(1) & df.is_employed.eq(1)].copy()
    df["salary_usd"] = pd.to_numeric(df.salary_usd, errors="coerce")
    df = df.loc[df.salary_usd.between(1_000, 500_000)].copy()
    df = df.loc[~df.age.astype(str).str.contains("under 18", case=False, na=False)].copy()
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL + SKILL_COLS:
        df[col] = df[col].fillna("").astype(str)
    return df.reset_index(drop=True)


def token_list(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(";") if x.strip()]


def make_preprocessor(with_skills: bool) -> ColumnTransformer:
    transformers = [
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ]
    if with_skills:
        for col in SKILL_COLS:
            # Vocabulary is learned only from the 2025 training data.
            transformers.append((f"{col}_tokens", CountVectorizer(
                tokenizer=token_list, token_pattern=None, lowercase=False,
                binary=True, min_df=25), col))
    return ColumnTransformer(transformers, remainder="drop")


def model_pipeline(with_skills: bool, params: dict | None = None) -> Pipeline:
    rf_params = {"n_estimators": 160, "max_depth": 20, "min_samples_split": 10,
                 "min_samples_leaf": 5, "max_features": 1.0, "max_samples": 0.8,
                 "random_state": SEED, "n_jobs": -1}
    if params:
        rf_params.update(params)
    forest = RandomForestRegressor(**rf_params)
    log_forest = TransformedTargetRegressor(
        regressor=forest, func=np.log1p, inverse_func=np.expm1,
        check_inverse=False)
    return Pipeline([("features", make_preprocessor(with_skills)), ("model", log_forest)])


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict:
    return {"MAE_USD": mean_absolute_error(y_true, prediction),
            "RMSE_USD": float(np.sqrt(mean_squared_error(y_true, prediction))),
            "Median_AE_USD": median_absolute_error(y_true, prediction),
            "R_squared": r2_score(y_true, prediction)}


def save_metrics_chart(frame: pd.DataFrame, path: Path) -> None:
    labels = frame["Model"].tolist()
    colors = ["#9aa6b2", "#6b8eaa", "#167d9a"][:len(labels)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    specs = [("MAE_USD", "Mean absolute error (lower is better)", True, "${:,.0f}"),
             ("RMSE_USD", "Root mean squared error (lower is better)", True, "${:,.0f}"),
             ("R_squared", "R² (higher is better)", False, "{:.3f}")]
    for ax, (key, title, lower, fmt) in zip(axes, specs):
        values = frame[key].to_numpy()
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", rotation=18, labelsize=8)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height(), fmt.format(value),
                    ha="center", va="bottom", fontsize=8)
        if key == "R_squared":
            ax.axhline(0, color="#555", linewidth=.8)
            lo = min(0, float(values.min()) - .05)
            ax.set_ylim(lo, max(.1, float(values.max()) * 1.18))
        else:
            ax.set_ylim(0, float(values.max()) * 1.22)
    fig.suptitle("Salary model evaluation — trained on 2025, tested on 2026", x=.02,
                 ha="left", fontsize=15, weight="bold")
    fig.text(.02, .91, "Lower error is better; higher R² is better. All salaries in USD.", color="#555")
    fig.tight_layout(rect=[0, 0, 1, .88])
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_actual_plot(y: pd.Series, pred: np.ndarray, path: Path) -> None:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(y), min(3500, len(y)), replace=False)
    actual, predicted = y.iloc[idx].to_numpy(), pred[idx]
    fig, ax = plt.subplots(figsize=(7.5, 7))
    ax.scatter(actual, predicted, s=10, alpha=.25, color="#167d9a", edgecolors="none")
    lim = [0, max(float(actual.max()), float(predicted.max()))]
    ax.plot(lim, lim, "--", color="#cf5c36", linewidth=1.5, label="Perfect prediction")
    ax.set(xlim=lim, ylim=lim, xlabel="Actual annual salary (USD)", ylabel="Predicted annual salary (USD)")
    ax.set_title("2026 holdout: predicted vs actual salaries", loc="left", weight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def get_importances(pipe: Pipeline) -> pd.DataFrame:
    prep = pipe.named_steps["features"]
    forest = pipe.named_steps["model"].regressor_
    names = prep.get_feature_names_out()
    imp = pd.DataFrame({"feature": names, "importance": forest.feature_importances_})
    # Display token names without their transformer prefix.
    imp["feature"] = imp.feature.str.replace(r"^.*__", "", regex=True)
    return imp.sort_values("importance", ascending=False).reset_index(drop=True)


def save_importance_chart(imp: pd.DataFrame, path: Path) -> None:
    top = imp.head(18).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.feature, top.importance, color="#167d9a")
    ax.set_title("Most influential model inputs", loc="left", weight="bold")
    ax.set_xlabel("Random forest feature importance")
    ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
    fig.text(.12, .01, "Importance describes model reliance; it does not show causation or skill payoff.", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, .04, 1, 1]); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def save_skill_sensitivity(pipe: Pipeline, test: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows = []
    for skill in SKILLS_TO_COMPARE:
        if skill == "java": col = "lang_list"
        elif skill == "amazon web services (aws)": col = "cloud_list"
        else: col = "cloud_list"
        present = test[col].apply(lambda s: skill.lower() in {x.lower() for x in token_list(s)})
        subset = test.loc[~present].copy()
        if subset.empty:
            continue
        with_skill = subset.copy()
        with_skill[col] = with_skill[col].apply(lambda s: (s + ";" if s else "") + skill)
        base_pred = pipe.predict(subset)
        changed_pred = pipe.predict(with_skill)
        delta = changed_pred - base_pred
        rows.append({"skill": skill, "mean_change_usd": float(np.mean(delta)),
                     "median_change_usd": float(np.median(delta)),
                     "profiles_with_increase_pct": float(np.mean(delta > 0) * 100),
                     "profiles_tested": len(delta)})
    result = pd.DataFrame(rows).sort_values("mean_change_usd")
    result.to_csv(OUT / "candidate_skill_sensitivity.csv", index=False)
    if not result.empty:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        colors = ["#c56b55" if x < 0 else "#167d9a" for x in result.mean_change_usd]
        bars = ax.barh(result.skill, result.mean_change_usd, color=colors)
        ax.axvline(0, color="#555", linewidth=.8)
        for bar, val in zip(bars, result.mean_change_usd):
            ax.text(val + (120 if val >= 0 else -120), bar.get_y()+bar.get_height()/2,
                    f"${val:+,.0f}", va="center", ha="left" if val >= 0 else "right", fontsize=9)
        ax.set_title("Model sensitivity to adding one skill", loc="left", weight="bold")
        ax.set_xlabel("Mean change in predicted annual salary (USD)")
        ax.grid(axis="x", alpha=.2); ax.set_axisbelow(True)
        fig.text(.12, .01, "Each skill is toggled on for 2026 profiles that lack it; other answers stay fixed. Not causal.", fontsize=8.5, color="#555")
        fig.tight_layout(rect=[0, .05, 1, 1]); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)
    return result


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    train = load_cohort(ROOT / "2025" / "stage1_2025.csv")
    test = load_cohort(ROOT / "2026" / "stage1_2026.csv")
    X_train, y_train = train[NUMERIC + CATEGORICAL + SKILL_COLS], train.salary_usd
    X_test, y_test = test[NUMERIC + CATEGORICAL + SKILL_COLS], test.salary_usd

    search_space = {
        "model__regressor__n_estimators": [120, 160, 220],
        "model__regressor__max_depth": [12, 20, None],
        "model__regressor__min_samples_split": [5, 10, 20],
        "model__regressor__min_samples_leaf": [3, 5, 10],
        "model__regressor__max_features": [0.7, 1.0],
        "model__regressor__max_samples": [0.7, 0.8, 0.9],
    }
    search = RandomizedSearchCV(
        model_pipeline(True), search_space, n_iter=10,
        scoring="neg_mean_absolute_error", cv=KFold(3, shuffle=True, random_state=SEED),
        random_state=SEED, n_jobs=1, verbose=1, refit=True, return_train_score=False)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    params = search.best_params_

    # Skill ablation uses the same tuned forest settings, changing only input features.
    no_skill = model_pipeline(False, {k.split("__")[-1]: v for k, v in params.items() if k.startswith("model__regressor__")})
    no_skill.fit(X_train, y_train)
    pred_base = no_skill.predict(X_test)
    pred_skill = best.predict(X_test)

    baseline = np.full(len(y_test), y_train.median())
    records = []
    for name, pred in [("2025 median baseline", baseline), ("Random forest (no skills)", pred_base),
                       ("Random forest (with skills)", pred_skill)]:
        records.append({"Model": name, **metrics(y_test, pred)})
    metric_df = pd.DataFrame(records)
    metric_df.to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame({"actual_salary_usd": y_test, "predicted_no_skills_usd": pred_base,
                  "predicted_with_skills_usd": pred_skill}).to_csv(OUT / "holdout_predictions.csv", index=False)
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(OUT / "hyperparameter_search_results.csv", index=False)
    (OUT / "best_parameters.json").write_text(json.dumps(search.best_params_, indent=2, default=str))
    importance = get_importances(best)
    importance.to_csv(OUT / "skill_feature_importances.csv", index=False)

    save_metrics_chart(metric_df, OUT / "model_metrics_comparison.png")
    save_actual_plot(y_test, pred_skill, OUT / "predicted_vs_actual.png")
    save_importance_chart(importance, OUT / "top_feature_importances.png")
    sensitivity = save_skill_sensitivity(best, X_test, OUT / "candidate_skill_sensitivity.png")

    print(f"Eligible rows — 2025 training: {len(train):,}; 2026 testing: {len(test):,}")
    print(f"Best 2025 cross-validation MAE: ${-search.best_score_:,.0f}")
    print("Best parameters:", search.best_params_)
    print("\n2026 holdout metrics:")
    print(metric_df.to_string(index=False, formatters={k: (lambda x: f"${x:,.0f}") for k in ["MAE_USD", "RMSE_USD", "Median_AE_USD"]} | {"R_squared": lambda x: f"{x:.4f}"}))
    if not sensitivity.empty:
        print("\nSkill sensitivity (model scenario, not causal):")
        print(sensitivity.to_string(index=False, formatters={"mean_change_usd": lambda x: f"${x:+,.0f}", "median_change_usd": lambda x: f"${x:+,.0f}"}))
    print(f"\nOutputs: {OUT}")


if __name__ == "__main__":
    main()
