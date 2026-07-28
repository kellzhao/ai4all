# Model directory

Drop the trained, exported model here as **`model.joblib`** (a scikit-learn or
XGBoost estimator saved with `joblib.dump(model, "model/model.joblib")`).

`feature_names.json` already lists the 21 BRFSS columns in the exact order
the model was trained on (`X = df.drop(columns=['Diabetes_binary'])`). The
app builds its input row using this order, so if the final training script
drops/reorders/adds columns, update this file to match.

The app works without a model present — it will show a placeholder message
in the UI instead of crashing — so the Streamlit/Docker side can be built
and reviewed before the final `.pkl`/`.joblib` lands here.

Whoever exports the final model, make sure it's one of the tree-based
models (RandomForestClassifier / XGBClassifier) so `shap.TreeExplainer`
works out of the box in `app/explain.py`. If you export a
LogisticRegression pipeline instead, swap `TreeExplainer` for
`shap.LinearExplainer` there.
