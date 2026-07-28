"""SHAP explanation for a single input row."""
import matplotlib.pyplot as plt
import shap
import streamlit as st


@st.cache_resource
def get_explainer(_model):
    """Cached TreeExplainer. Prefixing the arg with `_` tells st.cache_resource
    not to hash the (unhashable) model object itself."""
    return shap.TreeExplainer(_model)


def waterfall_figure(model, input_row):
    """Returns a matplotlib Figure with a SHAP waterfall plot for one row.

    Assumes a tree-based binary classifier (RandomForest/XGBoost). For a
    LogisticRegression model, swap `get_explainer` to `shap.LinearExplainer`.
    """
    explainer = get_explainer(model)
    explanation = explainer(input_row)

    # Binary classifiers: TreeExplainer returns shap values per class.
    # Take the "positive" (diabetes) class.
    if len(explanation.shape) == 3:
        explanation = explanation[:, :, 1]

    fig = plt.figure()
    shap.plots.waterfall(explanation[0], show=False)
    plt.tight_layout()
    return fig


def summarize_top_drivers(model, input_row, top_n=3):
    """Plain-English sentence naming the top features pushing risk up/down."""
    explainer = get_explainer(model)
    explanation = explainer(input_row)
    if len(explanation.shape) == 3:
        explanation = explanation[:, :, 1]

    values = explanation[0].values
    names = input_row.columns
    pairs = sorted(zip(names, values), key=lambda p: abs(p[1]), reverse=True)[:top_n]

    increasing = [n for n, v in pairs if v > 0]
    decreasing = [n for n, v in pairs if v < 0]

    parts = []
    if increasing:
        parts.append(f"raised primarily by {', '.join(increasing)}")
    if decreasing:
        parts.append(f"lowered by {', '.join(decreasing)}")
    return " and ".join(parts) if parts else "close to the model's baseline risk"
