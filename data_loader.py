import pandas as pd
import numpy as np

def load_data():
    df = pd.read_excel("covid19.xlsx")
    
    # Data preprocessing steps
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    cols_to_drop = missing_percentage[missing_percentage > 80].index.tolist()
    df = df.drop(columns=cols_to_drop)
    
    current_year = pd.Timestamp.now().year
    df['Age'] = current_year - df['Birth Year']
    
    cat_cols = [col for col in df.columns if df[col].dtype == 'object' or str(df[col].dtype) == 'category']
    df[cat_cols] = df[cat_cols].fillna("UNKNOWN")
    
    median_age = df['Age'].median()
    df['Age'] = df['Age'].fillna(median_age)
    
    df.drop(columns=['Birth Year'], inplace=True)
    df['Sex'] = df['Sex'].replace("OTHER", "UNKNOWN")
    df['Result'] = df['Result'].apply(lambda x: 'POSITIVE' if x == 'POSITIVE' else 'NOT_POSITIVE')
    df['Result'] = df['Result'].map({'NOT_POSITIVE': 0, 'POSITIVE': 1})
    
    return df

def get_preprocessed_data():
    df = load_data()
    
    # For prediction page - prepare encoded data
    X = df.drop(columns=['Result'])
    y = df['Result']
    
    # One-hot encode categorical variables
    cat_cols = [col for col in X.columns if X[col].dtype == 'object' or str(X[col].dtype) == 'category']
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=False).astype(int)
    
    return X_encoded, y, X.columns