# Salary analysis workbook

This folder contains a self-contained Jupyter notebook, preserved raw survey inputs, harmonized datasets, and analysis outputs. No inputs or project code outside this folder are required.

## Files

- `unharmonized_raw_data/2025_raw.csv` and `2026_raw.csv` — original survey downloads
- `unharmonized_raw_data/2025_schema.csv` and `2026_schema.csv` — source schema references
- `2025.csv` and `2026.csv` — harmonized datasets with the same columns and order
- `harmonized_schema.csv` — shared field names, source mappings, and descriptions (created by the notebook)
- `salary_analysis_workbook.ipynb` — preprocessing, descriptive, predictive, and prescriptive analytics
- `requirements.txt` — Python packages used by the notebook

## Run

Install the packages listed in `requirements.txt`, open `salary_analysis_workbook.ipynb` in Jupyter or Google Colab, and run all cells from top to bottom. In Colab, upload both raw CSVs directly beside the notebook using either `results.csv` + `survey_results_public.csv` or `2025_raw.csv` + `2026_raw.csv`; no input subfolder is required. When run locally, the notebook reads the preserved files from `unharmonized_raw_data/`. It harmonizes 2026 first and then 2025, writes the shared-schema datasets and schema reference to the working directory, and creates `outputs/` for tables and charts.

The predictive section uses previously cross-validated fixed random forest settings and fits linear regression on 2025, then compares both on 2026; routine runs do not repeat randomized search. The notebook also compares those regression predictions mapped to 2025-derived salary quartiles against direct random forest and logistic regression band classifiers on the 2026 holdout. The prescriptive analytics includes (1) a population-level one-skill-at-a-time sensitivity comparison and (2) a static profile career trajectory that cumulatively adds Java, Docker, Kubernetes, Terraform, AWS, GCP, and Azure, with experience increased at the three specified annual checkpoints. These are model-based what-if estimates, not causal effects. The notebook compares continuous-salary regression metrics (MAE, RMSE, median absolute error, and R²) and a separate four-band classification task (accuracy, balanced accuracy, precision/recall/F1, and confusion matrices).


## Project report

- `Salary_Analysis_Term_Project_Report.docx` — formatted term-project report with the analysis flow, figures, model results, team details, and bibliography.
