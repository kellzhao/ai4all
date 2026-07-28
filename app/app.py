import pandas as pd
import streamlit as st

from explain import summarize_top_drivers, waterfall_figure
from model_io import load_feature_names, load_model

st.set_page_config(page_title="XAI Diabetes Health Risk Reporter", layout="wide")

YES_NO = {"No": 0, "Yes": 1}

GENHLTH = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}

AGE_GROUPS = {
    "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
    "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
    "70-74": 11, "75-79": 12, "80+": 13,
}

EDUCATION = {
    "Never attended / kindergarten only": 1,
    "Elementary (grades 1-8)": 2,
    "Some high school (grades 9-11)": 3,
    "High school graduate / GED": 4,
    "Some college / technical school": 5,
    "College graduate": 6,
}

INCOME = {
    "Less than $10,000": 1, "$10,000-$14,999": 2, "$15,000-$19,999": 3,
    "$20,000-$24,999": 4, "$25,000-$34,999": 5, "$35,000-$49,999": 6,
    "$50,000-$74,999": 7, "$75,000 or more": 8,
}


def yes_no(label, help_text=None):
    return YES_NO[st.sidebar.radio(label, list(YES_NO), horizontal=True, help=help_text)]


def collect_inputs():
    st.sidebar.header("Your health profile")

    st.sidebar.subheader("Vitals")
    bmi = st.sidebar.number_input("BMI", min_value=12.0, max_value=98.0, value=27.0, step=0.1)
    genhlth = GENHLTH[st.sidebar.select_slider("General health", list(GENHLTH), value="Good")]
    menthlth = st.sidebar.slider("Poor mental health (days in past 30)", 0, 30, 0)
    physhlth = st.sidebar.slider("Poor physical health (days in past 30)", 0, 30, 0)

    st.sidebar.subheader("Conditions")
    highbp = yes_no("High blood pressure")
    highchol = yes_no("High cholesterol")
    cholcheck = yes_no("Cholesterol check in past 5 years")
    stroke = yes_no("Ever had a stroke")
    heart = yes_no("Heart disease or heart attack history")
    diffwalk = yes_no("Serious difficulty walking / climbing stairs")

    st.sidebar.subheader("Lifestyle")
    smoker = yes_no("Smoked 100+ cigarettes in your life")
    physactivity = yes_no("Physical activity in past 30 days (outside of job)")
    fruits = yes_no("Eat fruit 1+ times per day")
    veggies = yes_no("Eat vegetables 1+ times per day")
    hvyalcohol = yes_no("Heavy alcohol consumption")

    st.sidebar.subheader("Access to care")
    healthcare = yes_no("Has any healthcare coverage")
    nodoc = yes_no("Couldn't see a doctor in past year due to cost")

    st.sidebar.subheader("Demographics")
    sex = 1 if st.sidebar.radio("Sex", ["Female", "Male"], horizontal=True) == "Male" else 0
    age = AGE_GROUPS[st.sidebar.selectbox("Age group", list(AGE_GROUPS))]
    education = EDUCATION[st.sidebar.selectbox("Education level", list(EDUCATION))]
    income = INCOME[st.sidebar.selectbox("Household income", list(INCOME))]

    return {
        "HighBP": highbp, "HighChol": highchol, "CholCheck": cholcheck, "BMI": bmi,
        "Smoker": smoker, "Stroke": stroke, "HeartDiseaseorAttack": heart,
        "PhysActivity": physactivity, "Fruits": fruits, "Veggies": veggies,
        "HvyAlcoholConsump": hvyalcohol, "AnyHealthcare": healthcare,
        "NoDocbcCost": nodoc, "GenHlth": genhlth, "MentHlth": menthlth,
        "PhysHlth": physhlth, "DiffWalk": diffwalk, "Sex": sex, "Age": age,
        "Education": education, "Income": income,
    }


def main():
    st.title("XAI Diabetes Health Risk Reporter")
    st.caption(
        "Educational tool trained on CDC BRFSS survey data — not a medical "
        "diagnosis. Talk to a healthcare provider about your actual risk."
    )

    model = load_model()
    feature_names = load_feature_names()
    inputs = collect_inputs()

    if model is None:
        st.warning(
            "No model has been exported yet. Drop the trained "
            "`model.joblib` into the `model/` directory (see `model/README.md`) "
            "to enable predictions."
        )
        st.dataframe(pd.DataFrame([inputs]))
        return

    if st.sidebar.button("Calculate risk", type="primary"):
        row = pd.DataFrame([inputs])[feature_names]
        proba = model.predict_proba(row)[0, 1]

        st.metric("Estimated probability of diabetes / prediabetes", f"{proba:.0%}")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("What's driving this score")
            fig = waterfall_figure(model, row)
            st.pyplot(fig, clear_figure=True)
        with col2:
            st.subheader("In plain English")
            drivers = summarize_top_drivers(model, row)
            st.write(f"Your estimated risk is {proba:.0%}, {drivers}.")
            st.caption(
                "This reflects patterns in survey data, not a clinical "
                "diagnosis. Lifestyle factors here (activity, diet, BMI) "
                "are often the most actionable."
            )
    else:
        st.info("Fill in the sidebar and click **Calculate risk**.")


if __name__ == "__main__":
    main()
