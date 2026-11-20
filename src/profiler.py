"""
Step 1: Data Profiling and Quality Audit Module.
"""
import numpy as np
import pandas as pd

def audit_structure_and_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """Profiles column names, data types, non-null counts, and missing percentages."""
    profile = pd.DataFrame({
        "Column_Name": df.columns,
        "Inferred_Dtype": df.dtypes.astype(str),
        "Non_Null_Count": df.count().values,
        "Null_Count": df.isnull().sum().values,
        "Null_Percentage (%)": (df.isnull().mean() * 100).round(2).values
    })
    return profile

def audit_duplicates_and_keys(df: pd.DataFrame, primary_key: str = "CustomerID") -> dict:
    """Evaluates exact row duplicates and candidate primary key uniqueness."""
    exact_dupes = df.duplicated().sum()
    pk_dupes = df[primary_key].duplicated().sum() if primary_key in df.columns else None
    unique_keys = df[primary_key].nunique() if primary_key in df.columns else None
    
    return {
        "exact_duplicates": exact_dupes,
        "primary_key": primary_key,
        "primary_key_duplicates": pk_dupes,
        "unique_primary_keys": unique_keys,
        "total_records": len(df),
        "is_pk_valid": (unique_keys == len(df)) if unique_keys is not None else False
    }

def audit_domain_constraints(df: pd.DataFrame) -> dict:
    """Audits business rules and impossible values (negative charges, tenure, ticket counts)."""
    checks = {}
    if "TenureMonths" in df.columns:
        checks["Negative Tenure (< 0)"] = int((df["TenureMonths"] < 0).sum())
    if "MonthlyCharges" in df.columns:
        checks["Non-Positive Monthly Charges (<= 0)"] = int((df["MonthlyCharges"] <= 0).sum())
    if "SupportTickets" in df.columns:
        checks["Negative Support Tickets (< 0)"] = int((df["SupportTickets"] < 0).sum())
    
    # Arithmetic consistency
    if {"TenureMonths", "MonthlyCharges", "TotalCharges"}.issubset(df.columns):
        expected_total = (df["TenureMonths"] * df["MonthlyCharges"]).round(2)
        diff = (df["TotalCharges"].round(2) - expected_total).abs()
        checks["Total Charges Mismatch (|Actual - Expected| > 0.05)"] = int((diff > 0.05).sum())
        
    return checks

def audit_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Screens all continuous features using IQR (1.5 * IQR) and Z-Score (|Z| > 3.0) boundaries."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    results = []
    
    for col in num_cols:
        series = df[col]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        std = series.std(ddof=1)
        mean = series.mean()
        z_outliers = series[((series - mean).abs() / std) > 3.0] if std > 0 else []
        
        results.append({
            "Feature": col,
            "Min": series.min(),
            "Q1": round(q1, 2),
            "Median": round(series.median(), 2),
            "Q3": round(q3, 2),
            "Max": series.max(),
            "IQR": round(iqr, 2),
            "Lower_Bound": round(lower_bound, 2),
            "Upper_Bound": round(upper_bound, 2),
            "IQR_Outliers_Count": len(iqr_outliers),
            "Z_Score_Outliers_Count": len(z_outliers)
        })
        
    return pd.DataFrame(results)

