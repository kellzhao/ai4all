"""Loading the exported model and its feature contract."""
import json
from pathlib import Path

import joblib
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"


@st.cache_resource
def load_feature_names() -> list[str]:
    with open(FEATURE_NAMES_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_model():
    """Returns the estimator, or None if no model has been dropped in yet."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)
