
import pandas as pd

def clean_data(input_path: str) -> pd.DataFrame:
    """
    Load raw CSV, clean it, and return a cleaned DataFrame.
    """

    # 1. Load CSV
    df = pd.read_csv(input_path)
    print(f"Number of rows before cleaning: {df.shape[0]} and columns: {df.shape[1]}")

    # 2. Clean purchase_amount
    df["purchase_amount"] = pd.to_numeric(df["purchase_amount"], errors="coerce")
    df = df.dropna(subset=["purchase_amount"])
    df = df[df["purchase_amount"] > 0]

    # 3. Standardize country names
    df["country"] = df["country"].str.strip().str.lower()
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

    # 3b. Drop empty string countries
    df = df[df["country"].str.strip() != ""]

    # 3c. Drop NaN countries
    df = df.dropna(subset=["country"])

    # 4. Convert purchase_date to datetime and drop invalid
    df["purchase_date"] = pd.to_datetime(df["purchase_date"], errors="coerce")
    df = df.dropna(subset=["purchase_date"])

    # 5. Print after-cleaning stats
    print(f"Number of rows after cleaning: {df.shape[0]}")
    print(f"Unique countries after cleaning: {df['country'].unique()}")
    print(f"Unique purchase dates after cleaning: {df['purchase_date'].unique()}")

    return df


if __name__ == "__main__":
    # Run cleaner directly
    cleaned_df = clean_data("../data/raw_data.csv")
    # Save cleaned file
    cleaned_df.to_csv("../data/cleaned_data.csv", index=False)
    print("✅ Cleaned data saved to data/cleaned_data.csv")