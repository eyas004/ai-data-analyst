
import streamlit as st
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="/home/eyas/projects/study/3/.env")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
import io

st.title("AI Data Analyst")
st.write("Upload any CSV file and ask questions about it.")

# -----------------------
# FLEXIBLE CLEANING
# -----------------------
def clean_data(df: pd.DataFrame):

    rows_before = df.shape[0]

    # purchase_amount
    if "purchase_amount" in df.columns:
        df["purchase_amount"] = pd.to_numeric(df["purchase_amount"], errors="coerce")
        df = df.dropna(subset=["purchase_amount"])
        df = df[df["purchase_amount"] > 0]

    # country
    if "country" in df.columns:
        df["country"] = df["country"].astype(str).str.strip().str.lower()

        country_mapping = {
            "germany": "Germany",
            "ger": "Germany",
            "saudi arabia": "Saudi Arabia",
            "sau": "Saudi Arabia",
            "egypt": "Egypt",
            "egy": "Egypt",
            "jordan": "Jordan",
            "jor": "Jordan"
        }

        df["country"] = df["country"].map(country_mapping).fillna(df["country"])
        df = df[df["country"].str.strip() != ""]
        df = df.dropna(subset=["country"])

    # purchase_date
    if "purchase_date" in df.columns:
        df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
        df = df.dropna(subset=["purchase_date"])

    rows_after = df.shape[0]

    return df, rows_before, rows_after


# -----------------------
# FILE UPLOAD
# -----------------------
file = st.file_uploader("Upload CSV")

if file:

    df = pd.read_csv(file)

    df, before, after = clean_data(df)

    st.success(f"Cleaning complete. {before - after} rows removed.")

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # -----------------------
    # BUILD FLEXIBLE SUMMARY
    # -----------------------

    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()

    summary_parts = []

    summary_parts.append(f"Shape: {df.shape}")
    summary_parts.append(f"Columns: {df.columns.tolist()}")
    summary_parts.append(f"\nSample:\n{df.head().to_string()}")
    summary_parts.append(f"\nStats:\n{df.describe(include='all').to_string()}")
    summary_parts.append(f"\nInfo:\n{info_str}")

    # optional breakdowns
    if "country" in df.columns:
        summary_parts.append(f"\nCountry breakdown:\n{df['country'].value_counts().to_string()}")

    if "product_category" in df.columns:
        summary_parts.append(f"\nCategory breakdown:\n{df['product_category'].value_counts().to_string()}")
    if "product_id" in df.columns:
        summary_parts.append(f"\nProduct ID frequency:\n{df['product_id'].value_counts().to_string()}")

    summary = "\n".join(summary_parts)

    # -----------------------
    # QUESTION
    # -----------------------

    question = st.text_input("Ask a question about the data")

    if question:

        prompt = f"""
You are a data analyst.

You are given a dataset summary.

Answer the question ONLY using the data provided.

Rules:
- Do NOT write Python code
- Do NOT explain calculations
- Give the final answer directly with numbers

Dataset summary:
{summary}

Question:
{question}
"""

        response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
)
        st.write(response.choices[0].message.content)

        # st.subheader("AI Answer")
        # st.write(response["message"]["content"])
