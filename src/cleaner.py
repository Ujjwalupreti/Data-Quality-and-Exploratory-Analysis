"""
Step 2: Data Cleaning, Standardization, and Validation Pipeline Module.
"""
import pandas as pd
from .config import COLUMN_RENAME_MAP, CONTRACT_MAPPING

def clean_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Executes reproducible cleaning strategy:
    1. Standardizes headers to lower snake_case.
    2. Strips whitespace across text fields.
    3. Normalizes contract nomenclature (Monthly, Annual, Two-Year).
    4. Engineers numeric binary churn indicator `is_churned`.
    """
    df = df_raw.copy()
    
    # 1. Rename columns
    df.rename(columns=COLUMN_RENAME_MAP, inplace=True)
    
    # 2. Strip whitespace
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        
    # 3. Standardize contract labels
    if "contract_type" in df.columns:
        df["contract_type_standardized"] = df["contract_type"].replace(CONTRACT_MAPPING)
        
    # 4. Binary churn indicator
    if "churned" in df.columns:
        df["is_churned"] = (df["churned"] == "Yes").astype(int)
        
    return df

def validate_cleaned_dataset(df_clean: pd.DataFrame) -> bool:
    """
    Executes automated defensive assertions on the cleaned dataset.
    Raises AssertionError if any quality standard is breached.
    """
    assert df_clean.duplicated().sum() == 0, "Assertion Error: Duplicate rows found!"
    assert df_clean["customer_id"].is_unique, "Assertion Error: Primary key is not unique!"
    assert df_clean.isnull().sum().sum() == 0, "Assertion Error: Null cells detected!"
    assert (df_clean["tenure_months"] >= 0).all(), "Assertion Error: Negative tenure found!"
    assert (df_clean["monthly_charge"] > 0).all(), "Assertion Error: Non-positive monthly charges found!"
    assert (df_clean["support_tickets"] >= 0).all(), "Assertion Error: Negative support tickets found!"
    assert set(df_clean["churned"].unique()).issubset({"Yes", "No"}), "Assertion Error: Invalid churn categories!"
    assert set(df_clean["contract_type_standardized"].unique()).issubset({"Monthly", "Annual", "Two-Year"}), "Assertion Error: Invalid contract categories!"
    return True

