

# Diabetes Health Indicators - Exploratory Data Analysis (BRFSS 2015) AI4ALL Group 23E

**Live demo:** [ai4all-23e.streamlit.app](https://ai4all-23e.streamlit.app/)
**Github page:** (https://kellzhao.github.io/ai4all/)

## Project Summary

This project explores relationships between health indicators, demographics, and diabetes status using the 2015 CDC Behavioral Risk Factor Surveillance System (BRFSS) survey data (~253K respondents). We ran an exploratory data analysis and bias audit, then trained and compared Logistic Regression, Random Forest, and XGBoost classifiers on both a class-balanced and a real-world-imbalanced version of the data, tuning for recall since missing a true diabetic case is the costlier error in a screening context. The final XGBoost model (ROC-AUC 0.83) is deployed in a Streamlit app that returns a risk estimate along with a SHAP-based explanation of which health factors drove that individual's prediction.

---

## Repository Contents

**App (deployed demo)**
- `app/app.py` — Streamlit UI: collects a health profile and shows the predicted risk with a SHAP explanation
- `app/explain.py` — builds the SHAP waterfall chart and plain-English driver summary
- `app/model_io.py` — loads the trained model and feature list
- `Dockerfile`, `docker-compose.yml`, `requirements.txt` — containerized deployment config

**Model**
- `model/model.joblib` — trained XGBoost classifier
- `model/feature_names.json` — the 21 input features, in training order
- `xgboost_model.ipynb` — trains the final XGBoost model and sweeps decision thresholds
- `model_comparison_results.csv` — accuracy/recall/precision/F1/ROC-AUC for Logistic Regression, Random Forest, and XGBoost on both balanced and imbalanced data
- `diabetes_model_comparison.ipynb` — the full model comparison and hyperparameter search behind that results table
- `diabetes_model_basic_kelly.ipynb`, `diabetes_model_edits_kelly .ipynb`, `model_1.py`, `model_2.py` — earlier modeling iterations

**Exploratory data analysis**
- `eda_bias_audit.ipynb` — data quality checks, correlations, and a bias audit of diabetes rate by sex/age/education/income
- `eda-visualizations.ipynb` — generates the four EDA charts (correlation heatmap, top predictors, BMI × age, income/education vs. diabetes rate)
- `*.png` — the exported chart images used in the analysis and presentation

**Data**
- `diabetes_binary_health_indicators_BRFSS2015.csv` — full dataset, real-world class imbalance (~86%/14%)
- `diabetes_binary_5050split_health_indicators_BRFSS2015.csv` — class-balanced 50/50 version
- `diabetes_012_health_indicators_BRFSS2015.csv` — 3-class version (no diabetes / prediabetes / diabetes)
