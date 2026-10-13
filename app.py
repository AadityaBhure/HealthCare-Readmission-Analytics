"""Streamlit entry point:  streamlit run app.py

Routes the five dashboard pages with st.navigation and renders the shared sidebar.
The same file is used locally and on Streamlit Community Cloud.
"""
import streamlit as st

from src import config, dashboard_data as dd, ui

st.set_page_config(page_title=config.APP_TITLE, page_icon="assets/icon.svg", layout="wide",
                   initial_sidebar_state="expanded")
ui.inject_css()
st.logo("assets/logo.svg", size="large", icon_image="assets/icon.svg")

pages = {
    "Analytics": [
        st.Page("pages/1_Overview.py", title="California Overview", icon=":material/dashboard:", default=True),
        st.Page("pages/2_CABG_Prediction.py", title="CABG Hospital Prediction", icon=":material/cardiology:",
                url_path="cabg-prediction"),
        st.Page("pages/3_Stroke_Analytics.py", title="Ischemic Stroke Analytics", icon=":material/neurology:",
                url_path="stroke-analytics"),
        st.Page("pages/4_Population_Trends.py", title="Population & County Trends", icon=":material/groups:",
                url_path="population-trends"),
    ],
    "Technical": [
        st.Page("pages/5_Model_Evaluation.py", title="Model Evaluation & Testing", icon=":material/science:",
                url_path="model-evaluation"),
    ],
}
nav = st.navigation(pages)

status = dd.ensure_artifacts()
schema = dd.schema_json("cabg")
with st.sidebar:
    st.markdown(
        f"""<div class="sidebar-note">
        <b style="color:#E6EDF7">Data coverage</b><br>
        CABG hospital measures: 2018-2023 (two-year windows)<br>
        Ischemic stroke outcomes: 2011-2015 (two-year windows)<br>
        All-cause readmissions: 2011-2024 (ICD-9 to 2015, ICD-10 from 2016)<br><br>
        <b style="color:#E6EDF7">CABG model in use</b><br>
        {schema['models'][schema['recommended_model']]} (selected by hospital-grouped CV on training data)<br>
        Trained {schema['created_utc'][:10]} - scikit-learn {schema['sklearn_version']}<br><br>
        <b style="color:#E6EDF7">Responsible use</b><br>
        Aggregate hospital/population statistics only. Estimates are hospital-level rates, not patient
        probabilities, and are not clinically validated.<br><br>
        <span style="opacity:.75">Status: {status}</span>
        </div>""",
        unsafe_allow_html=True)

nav.run()
