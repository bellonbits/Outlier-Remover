import streamlit as st
import pandas as pd
import numpy as np

# Function to detect outliers using IQR
def detect_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
    return outliers

# Function to clean outliers by removing them
def clean_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    cleaned_df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return cleaned_df

# Streamlit app
st.title("Outlier Detection and Removal")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Original Dataset")
    st.write(df)
    
    # Detect outliers
    st.write("Detecting outliers in numerical columns...")
    outliers = detect_outliers(df.select_dtypes(include=[np.number]))
    st.write(outliers)
    
    # Show rows with outliers
    st.write("Rows with outliers:")
    st.write(df[outliers.any(axis=1)])

    # Clean outliers
    st.write("Cleaning outliers...")
    cleaned_df = clean_outliers(df.select_dtypes(include=[np.number]))
    st.write("Dataset after removing outliers:")
    st.write(cleaned_df)
    
    # Download cleaned dataset
    cleaned_csv = cleaned_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download cleaned CSV", data=cleaned_csv, file_name="cleaned_dataset.csv", mime="text/csv")
else:
    st.write("Please upload a CSV file to begin.")
