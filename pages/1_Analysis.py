import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from data_loader import load_data

# Load data
df = load_data()

# Page configuration
st.set_page_config(layout="wide")
st.title("COVID-19 Data Analysis Dashboard")
st.markdown("""
<style>
.big-font {
    font-size:18px !important;
}
</style>
""", unsafe_allow_html=True)

# Section 1: Dataset Overview
st.header("1. Dataset Overview")
col1, col2, col3 = st.columns([1,1,2])
with col1:
    st.metric("Total Records", df.shape[0])
with col2:
    st.metric("Total Features", df.shape[1])
with col3:
    if st.checkbox("View Sample Data"):
        st.dataframe(df.head(5).style.highlight_max(axis=0))

# Section 2: Target Distribution
st.header("2. Target Variable Distribution")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(8,4))
    sns.countplot(x=df['Result'], palette="viridis")
    plt.title("COVID-19 Test Results")
    plt.xlabel("Result (0=Negative, 1=Positive)")
    plt.ylabel("Count")
    st.pyplot(fig)
    plt.close()
with col2:
    fig, ax = plt.subplots(figsize=(8,4))
    df['Result'].value_counts().plot.pie(autopct='%1.1f%%', 
                                       colors=['#4CAF50','#F44336'],
                                       startangle=90)
    plt.title("Result Percentage Distribution")
    st.pyplot(fig)
    plt.close()

# Section 3: Missing Values Analysis
st.header("3. Data Quality Analysis")
missing_percentage = (df.isnull().sum() / len(df)) * 100
missing_percentage = missing_percentage[missing_percentage > 0]

if not missing_percentage.empty:
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(10,4))
        missing_percentage.sort_values(ascending=False).plot.bar(color='#FF9800')
        plt.title("Missing Values Percentage")
        plt.ylabel("Percentage Missing")
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
    with col2:
        st.dataframe(missing_percentage.round(2).rename("Missing %").to_frame())
else:
    st.success("No missing values in the dataset")

# Section 4: Age Analysis
st.header("4. Age Distribution Analysis")
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(df['Age'], bins=30, kde=True, color='#2196F3')
    plt.title("Age Distribution")
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(x=df['Age'], color='#00BCD4')
    plt.title("Age Boxplot")
    st.pyplot(fig)
    plt.close()

# Section 5: Categorical Features
st.header("5. Categorical Features Analysis")
cat_cols = [col for col in df.columns if df[col].dtype == 'object' or str(df[col].dtype) == 'category']

for col in cat_cols:
    st.subheader(f"Feature: {col}")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10,4))
        sns.countplot(y=df[col], order=df[col].value_counts().index, 
                     palette="rocket")
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.dataframe(df[col].value_counts().rename("Count").to_frame())