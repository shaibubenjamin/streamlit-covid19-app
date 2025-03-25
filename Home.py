import streamlit as st

# Configure the app (only in main app.py)
st.set_page_config(
    page_title="COVID-19 Prediction App",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Home page content
st.title("🦠 COVID-19 Analysis & Prediction")
st.write("""Welcome to the COVID-19 prediction system.""")