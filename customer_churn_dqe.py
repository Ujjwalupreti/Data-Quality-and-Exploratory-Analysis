"""
Data Quality and Exploratory Analysis (DQE) Pipeline
Dataset: customer_churn_sample.csv -> customer_churn_cleaned.csv
Author: Senior Data Analyst
Target Project: EDAI - Data Analytics Portfolio Project
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for publication-ready visualisations
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.autolayout": True
})

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, "customer_churn_sample.csv")
CLEAN_DATA_PATH = os.path.join(SCRIPT_DIR, "customer_churn_cleaned.csv")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("DATA QUALITY AND EXPLORATORY ANALYSIS (DQE) PIPELINE")
print("=" * 80)

# ==============================================================================
# STEP 1: DATA PROFILING (RAW DATA AUDIT)
# ==============================================================================
print("\n" + "=" * 50)
print("STEP 1: DATA PROFILING & QUALITY AUDIT")
print("=" * 50)

# 1.1 Load Raw Dataset
if not os.path.exists(RAW_DATA_PATH):
    raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_PATH}")

df_raw = pd.read_csv(RAW_DATA_PATH)
n_rows, n_cols = df_raw.shape
print(f"[Profiling] Raw Dataset Loaded: {n_rows} rows x {n_cols} columns")
print(f"[Profiling] Columns Found: {list(df_raw.columns)}")

# 1.2 Structure & Data Types
print("\n--- 1.2 Data Types & Non-Null Counts ---")
profile_dtypes = pd.DataFrame({
    "Column": df_raw.columns,
    "Data_Type": df_raw.dtypes.astype(str),
    "Non_Null_Count": df_raw.count().values,
    "Null_Count": df_raw.isnull().sum().values,
    "Null_Pct": (df_raw.isnull().mean() * 100).round(2).values
})
print(profile_dtypes.to_string(index=False))

# 1.3 Missing Value Audit
missing_total = df_raw.isnull().sum().sum()
print(f"\n--- 1.3 Missing Value Audit ---")
print(f"Total Missing Cells across entire dataset: {missing_total}")
if missing_total == 0:
    print("Result: No missing values detected in the raw dataset.")
else:
    print(df_raw.isnull().sum()[df_raw.isnull().sum() > 0])

# 1.4 Duplicate Rows & Duplicate Identifiers
print("\n--- 1.4 Duplicate Rows & Primary Key Audit ---")
exact_duplicates = df_raw.duplicated().sum()
id_col_raw = [c for c in df_raw.columns if "id" in c.lower()][0]
id_duplicates = df_raw[id_col_raw].duplicated().sum()
n_unique_ids = df_raw[id_col_raw].nunique()
print(f"Exact Duplicate Rows: {exact_duplicates}")
print(f"Duplicate '{id_col_raw}' (Primary Key) values: {id_duplicates}")
print(f"Unique '{id_col_raw}' Count: {n_unique_ids} out of {n_rows} total rows")

# 1.5 Impossible / Domain Constraint Checks
print("\n--- 1.5 Domain Constraint & Impossible Value Audit ---")
# Check numerical columns
num_cols_raw = df_raw.select_dtypes(include=[np.number]).columns.tolist()
impossible_findings = []

for col in num_cols_raw:
    neg_count = (df_raw[col] < 0).sum()
    if neg_count > 0:
        impossible_findings.append(f"Negative values in {col}: {neg_count} rows")

# Domain checks on tenure and charges
if "TenureMonths" in df_raw.columns:
    inv_tenure = (df_raw["TenureMonths"] < 0).sum()
    print(f"Impossible Tenure (< 0): {inv_tenure} rows")
if "MonthlyCharges" in df_raw.columns:
    inv_charges = (df_raw["MonthlyCharges"] <= 0).sum()
    print(f"Impossible Monthly Charges (<= 0): {inv_charges} rows")
if "SupportTickets" in df_raw.columns:
    inv_tickets = (df_raw["SupportTickets"] < 0).sum()
    print(f"Impossible Support Tickets (< 0): {inv_tickets} rows")

# Total charges consistency check (TenureMonths * MonthlyCharges ~ TotalCharges)
if {"TenureMonths", "MonthlyCharges", "TotalCharges"}.issubset(df_raw.columns):
    expected_total = (df_raw["TenureMonths"] * df_raw["MonthlyCharges"]).round(2)
    diff = (df_raw["TotalCharges"].round(2) - expected_total).abs()
    mismatch_count = (diff > 0.05).sum()
    print(f"Total Charges calculation mismatches (|Actual - Expected| > 0.05): {mismatch_count}")

# Categorical domain check
if "ContractType" in df_raw.columns:
    print(f"ContractType Unique Values: {df_raw['ContractType'].unique().tolist()}")
if "Churn" in df_raw.columns:
    print(f"Churn Unique Values: {df_raw['Churn'].unique().tolist()}")

# 1.6 Outlier Detection (IQR & Z-Score)
print("\n--- 1.6 Outlier Detection (IQR & Z-Score Analysis) ---")
outlier_summary = []
for col in num_cols_raw:
    series = df_raw[col]
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    iqr_outliers = series[(series < lower_bound) | (series > upper_bound)]
    
    # Z-score (sample std with ddof=1)
    std = series.std(ddof=1)
    mean = series.mean()
    if std > 0:
        z_scores = (series - mean).abs() / std
        z_outliers = series[z_scores > 3.0]
    else:
        z_outliers = pd.Series([], dtype=float)
        
    outlier_summary.append({
        "Feature": col,
        "Min": series.min(),
        "Q1 (25%)": round(q1, 2),
        "Median (50%)": round(series.median(), 2),
        "Q3 (75%)": round(q3, 2),
        "Max": series.max(),
        "IQR": round(iqr, 2),
        "Lower_Bound": round(lower_bound, 2),
        "Upper_Bound": round(upper_bound, 2),
        "IQR_Outliers_Count": len(iqr_outliers),
        "Z_Score_Outliers_Count (>3σ)": len(z_outliers)
    })

df_outliers = pd.DataFrame(outlier_summary)
print(df_outliers[["Feature", "Min", "Median (50%)", "Max", "Lower_Bound", "Upper_Bound", "IQR_Outliers_Count", "Z_Score_Outliers_Count (>3σ)"]].to_string(index=False))

# ==============================================================================
# STEP 2: DATA CLEANING & VALIDATION
# ==============================================================================
print("\n" + "=" * 50)
print("STEP 2: DATA CLEANING & POST-VALIDATION")
print("=" * 50)

df_clean = df_raw.copy()

# 2.1 Standardize Column Names to snake_case schema
# Standardizing schema:
# CustomerID -> customer_id
# TenureMonths -> tenure_months
# MonthlyCharges -> monthly_charge
# SupportTickets -> support_tickets
# ContractType -> contract_type
# Churn -> churned
col_rename_map = {
    "CustomerID": "customer_id",
    "Gender": "gender",
    "Age": "age",
    "TenureMonths": "tenure_months",
    "SubscriptionType": "subscription_type",
    "MonthlyCharges": "monthly_charge",
    "TotalCharges": "total_charges",
    "ContractType": "contract_type",
    "SupportTickets": "support_tickets",
    "PaymentMethod": "payment_method",
    "Churn": "churned"
}
df_clean.rename(columns=col_rename_map, inplace=True)
print(f"[Cleaning] Standardized column headers to lowercase snake_case:")
print(f"           {list(df_clean.columns)}")

# 2.2 Text / String Standardization
# Strip leading/trailing whitespaces from string columns
str_cols = df_clean.select_dtypes(include="object").columns
for c in str_cols:
    df_clean[c] = df_clean[c].astype(str).str.strip()

# Standardize contract_type nomenclature:
# 'Month-to-Month' -> 'Monthly'
# 'One Year' -> 'Annual'
# 'Two Year' -> 'Two-Year'
contract_mapping = {
    "Month-to-Month": "Monthly",
    "One Year": "Annual",
    "Two Year": "Two-Year"
}
df_clean["contract_type_standardized"] = df_clean["contract_type"].replace(contract_mapping)

# Add binary churn flag (1 for Yes, 0 for No) for numerical aggregation/BI tools
df_clean["is_churned"] = (df_clean["churned"] == "Yes").astype(int)

# 2.3 Post-Cleaning Automated Assertions (Defensive Validation)
print("\n--- 2.3 Executing Post-Cleaning Assertions ---")
assert df_clean.duplicated().sum() == 0, "Validation Failed: Exact duplicate rows detected!"
assert df_clean["customer_id"].is_unique, "Validation Failed: Customer ID is not unique!"
assert df_clean.isnull().sum().sum() == 0, "Validation Failed: Null values detected!"
assert (df_clean["tenure_months"] >= 0).all(), "Validation Failed: Negative tenure found!"
assert (df_clean["monthly_charge"] > 0).all(), "Validation Failed: Non-positive monthly charges found!"
assert (df_clean["support_tickets"] >= 0).all(), "Validation Failed: Negative support tickets found!"
assert set(df_clean["churned"].unique()).issubset({"Yes", "No"}), "Validation Failed: Invalid churn values!"
assert set(df_clean["contract_type_standardized"].unique()).issubset({"Monthly", "Annual", "Two-Year"}), "Validation Failed: Invalid contract types!"
print(">>> ALL ASSERTIONS PASSED! Dataset is clean, consistent, and validated.")

# 2.4 Export Cleaned Dataset
# Exporting the production dataset matching the required specification
export_columns = [
    "customer_id",
    "gender",
    "age",
    "tenure_months",
    "subscription_type",
    "monthly_charge",
    "total_charges",
    "contract_type",
    "contract_type_standardized",
    "support_tickets",
    "payment_method",
    "churned",
    "is_churned"
]
df_clean[export_columns].to_csv(CLEAN_DATA_PATH, index=False)
print(f"\n[Export] Cleaned dataset saved successfully to:\n         {CLEAN_DATA_PATH}")

# ==============================================================================
# STEP 3: SUMMARY STATISTICS & EXPLORATORY DISTRIBUTIONS
# ==============================================================================
print("\n" + "=" * 50)
print("STEP 3: SUMMARY STATISTICS & EXPLORATORY DISTRIBUTIONS")
print("=" * 50)

# 3.1 Numerical Summary Statistics (Excluding customer_id)
num_analysis_cols = ["tenure_months", "monthly_charge", "total_charges", "support_tickets", "age"]
stats_overall = df_clean[num_analysis_cols].describe(percentiles=[0.25, 0.50, 0.75]).T
stats_overall["median"] = df_clean[num_analysis_cols].median()
stats_overall["iqr"] = stats_overall["75%"] - stats_overall["25%"]
print("\n--- 3.1 Numerical Summary Statistics (Overall) ---")
print(stats_overall[["count", "mean", "std", "min", "25%", "median", "75%", "max", "iqr"]].to_string())

# 3.2 Categorical Frequencies & Percentages
print("\n--- 3.2 Categorical Frequencies ---")
for cat_col in ["contract_type_standardized", "subscription_type", "payment_method", "churned"]:
    freq_table = pd.DataFrame({
        "Frequency": df_clean[cat_col].value_counts(),
        "Percentage (%)": (df_clean[cat_col].value_counts(normalize=True) * 100).round(2)
    })
    print(f"\nFeature: {cat_col}")
    print(freq_table.to_string())

# 3.3 Segmented Statistics: Retained vs Churned
print("\n--- 3.3 Comparative Statistics by Churn Status ---")
comp_stats = df_clean.groupby("churned")[num_analysis_cols].agg(["mean", "median", "std"]).T
print(comp_stats.round(2).to_string())

# Crosstab: Contract Type vs Churn
print("\n--- 3.4 Crosstab: Contract Type vs Churn ---")
ct_crosstab = pd.crosstab(df_clean["contract_type_standardized"], df_clean["churned"], margins=True)
ct_crosstab_pct = pd.crosstab(df_clean["contract_type_standardized"], df_clean["churned"], normalize="index") * 100
print("Counts:")
print(ct_crosstab)
print("\nRow Percentages (%):")
print(ct_crosstab_pct.round(2))

# Crosstab: Support Tickets vs Churn
print("\n--- 3.5 Crosstab: Support Tickets vs Churn ---")
tickets_crosstab = pd.crosstab(df_clean["support_tickets"], df_clean["churned"], margins=True)
print(tickets_crosstab)

# ==============================================================================
# VISUALIZATION GENERATION
# ==============================================================================
print("\n" + "=" * 50)
print("GENERATING VISUALIZATIONS")
print("=" * 50)

# Color palette: Blue for Retained (No), Crimson/Coral for Churned (Yes)
palette = {"No": "#1f77b4", "Yes": "#d62728"}

# Figure 1: Churn Rate by Contract Type
fig, ax = plt.subplots(figsize=(8, 5))
contract_churn_pct = df_clean.groupby("contract_type_standardized")["is_churned"].mean() * 100
contract_counts = df_clean["contract_type_standardized"].value_counts()
bars = ax.bar(contract_churn_pct.index, contract_churn_pct.values, color=["#d62728" if v > 50 else "#1f77b4" for v in contract_churn_pct.values], width=0.55, edgecolor="black", alpha=0.85)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2.0, height + 2, f"{height:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.set_ylim(0, 115)
ax.set_title("Descriptive Finding: Churn Rate by Contract Type", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Contract Type", fontsize=12, labelpad=8)
ax.set_ylabel("Observed Churn Rate (%)", fontsize=12, labelpad=8)
ax.grid(axis="y", linestyle="--", alpha=0.7)
fig1_path = os.path.join(FIGURES_DIR, "01_churn_by_contract_type.png")
fig.savefig(fig1_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[Chart Saved] Figure 1: {fig1_path}")

# Figure 2: Support Tickets vs Churn Status
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df_clean, x="support_tickets", hue="churned", palette=palette, ax=ax, edgecolor="black", alpha=0.85)
ax.set_title("Distribution of 90-Day Support Tickets by Churn Status", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Support Tickets (Count in Prior 90 Days)", fontsize=12, labelpad=8)
ax.set_ylabel("Customer Count", fontsize=12, labelpad=8)
ax.legend(title="Churned", title_fontsize="11", loc="upper right")
ax.axvline(x=2.5, color="red", linestyle="--", linewidth=1.8, label="Ticket Inflection Line (≥3)")
ax.text(2.6, 3.5, "Sharp Inflection:\nAll customers with ≥3\ntickets churned (7/7)", color="darkred", fontsize=10, fontweight="semibold")
ax.grid(axis="y", linestyle="--", alpha=0.7)
fig2_path = os.path.join(FIGURES_DIR, "02_support_tickets_vs_churn.png")
fig.savefig(fig2_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[Chart Saved] Figure 2: {fig2_path}")

# Figure 3: Tenure Months Distribution by Churn Status
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df_clean, x="churned", y="tenure_months", hue="churned", palette=palette, ax=ax, width=0.45, boxprops=dict(alpha=0.85, edgecolor="black"), legend=False)
sns.stripplot(data=df_clean, x="churned", y="tenure_months", color="black", size=7, jitter=0.15, ax=ax, alpha=0.75)
ax.set_title("Tenure Months Distribution by Churn Status", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Customer Churned?", fontsize=12, labelpad=8)
ax.set_ylabel("Tenure (Completed Months)", fontsize=12, labelpad=8)
ax.grid(axis="y", linestyle="--", alpha=0.7)
fig3_path = os.path.join(FIGURES_DIR, "03_tenure_distribution_by_churn.png")
fig.savefig(fig3_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[Chart Saved] Figure 3: {fig3_path}")

# Figure 4: Monthly Charge vs Tenure by Churn Status
fig, ax = plt.subplots(figsize=(9, 5.5))
sns.scatterplot(
    data=df_clean,
    x="tenure_months",
    y="monthly_charge",
    hue="churned",
    style="contract_type_standardized",
    palette=palette,
    s=140,
    ax=ax,
    edgecolor="black",
    alpha=0.9
)
ax.set_title("Customer Segmentation: Tenure vs Monthly Charge (INR)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Tenure (Completed Months)", fontsize=12, labelpad=8)
ax.set_ylabel("Monthly Charge (INR)", fontsize=12, labelpad=8)
ax.legend(title="Churn / Contract", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.axvspan(0, 15, color="red", alpha=0.08, label="High-Risk Zone (Tenure < 15 mos)")
ax.grid(True, linestyle="--", alpha=0.6)
fig4_path = os.path.join(FIGURES_DIR, "04_monthly_charge_vs_tenure_scatter.png")
fig.savefig(fig4_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[Chart Saved] Figure 4: {fig4_path}")

# Figure 5: Multi-Panel Portfolio Dashboard Figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Customer Churn Data Quality & Exploratory Analysis Dashboard", fontsize=16, fontweight="bold", y=0.99)

# Panel 1: Churn by Contract
bars1 = axes[0, 0].bar(contract_churn_pct.index, contract_churn_pct.values, color=["#d62728" if v > 50 else "#1f77b4" for v in contract_churn_pct.values], width=0.5, edgecolor="black")
for bar in bars1:
    h = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width() / 2.0, h + 2, f"{h:.0f}%", ha="center", va="bottom", fontweight="bold")
axes[0, 0].set_ylim(0, 115)
axes[0, 0].set_title("A: Churn Rate by Contract Type", fontweight="bold", fontsize=12)
axes[0, 0].set_ylabel("Churn Rate (%)")
axes[0, 0].set_xlabel("Contract Type")
axes[0, 0].grid(axis="y", linestyle="--", alpha=0.6)

# Panel 2: Tickets by Churn
sns.countplot(data=df_clean, x="support_tickets", hue="churned", palette=palette, ax=axes[0, 1], edgecolor="black", alpha=0.85)
axes[0, 1].set_title("B: Support Tickets (Prior 90 Days) by Churn", fontweight="bold", fontsize=12)
axes[0, 1].set_xlabel("Support Ticket Count")
axes[0, 1].set_ylabel("Customer Count")
axes[0, 1].axvline(x=2.5, color="red", linestyle="--", linewidth=1.5)
axes[0, 1].legend(title="Churned", loc="upper right")
axes[0, 1].grid(axis="y", linestyle="--", alpha=0.6)

# Panel 3: Tenure Boxplot
sns.boxplot(data=df_clean, x="churned", y="tenure_months", hue="churned", palette=palette, ax=axes[1, 0], width=0.4, boxprops=dict(alpha=0.85, edgecolor="black"), legend=False)
sns.stripplot(data=df_clean, x="churned", y="tenure_months", color="black", size=6, jitter=0.15, ax=axes[1, 0], alpha=0.7)
axes[1, 0].set_title("C: Tenure Months Distribution by Churn", fontweight="bold", fontsize=12)
axes[1, 0].set_xlabel("Churned?")
axes[1, 0].set_ylabel("Tenure Months")
axes[1, 0].grid(axis="y", linestyle="--", alpha=0.6)

# Panel 4: Subscription Plan vs Churn
sub_churn_pct = df_clean.groupby("subscription_type")["is_churned"].mean() * 100
sub_order = ["Basic", "Pro", "Enterprise"]
sub_vals = [sub_churn_pct.get(k, 0) for k in sub_order]
bars4 = axes[1, 1].bar(sub_order, sub_vals, color=["#d62728", "#ff7f0e", "#2ca02c"], width=0.5, edgecolor="black", alpha=0.85)
for bar in bars4:
    h = bar.get_height()
    axes[1, 1].text(bar.get_x() + bar.get_width() / 2.0, h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
axes[1, 1].set_ylim(0, 100)
axes[1, 1].set_title("D: Churn Rate by Subscription Tier", fontweight="bold", fontsize=12)
axes[1, 1].set_xlabel("Subscription Tier")
axes[1, 1].set_ylabel("Churn Rate (%)")
axes[1, 1].grid(axis="y", linestyle="--", alpha=0.6)

fig5_path = os.path.join(FIGURES_DIR, "05_eda_summary_dashboard.png")
fig.savefig(fig5_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[Chart Saved] Figure 5: {fig5_path}")

print("\n" + "=" * 80)
print("DQE PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
