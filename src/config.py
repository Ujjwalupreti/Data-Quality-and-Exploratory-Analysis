"""
Configuration, path management, and plotting themes for DQE pipeline.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Base directories
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Data paths
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "customer_churn_sample.csv")
CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "customer_churn_cleaned.csv")
ROOT_CLEANED_DATA_PATH = os.path.join(PROJECT_ROOT, "customer_churn_cleaned.csv")

# Reports and figures
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
ROOT_FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(ROOT_FIGURES_DIR, exist_ok=True)

# Schema column mappings
COLUMN_RENAME_MAP = {
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

CONTRACT_MAPPING = {
    "Month-to-Month": "Monthly",
    "One Year": "Annual",
    "Two Year": "Two-Year"
}

# Color palettes (Colorblind safe)
PALETTE_CHURN = {"No": "#1f77b4", "Yes": "#d62728"}
PALETTE_TIERS = ["#d62728", "#ff7f0e", "#2ca02c"]

def init_visual_style():
    """Initializes standard publication-ready Matplotlib & Seaborn settings."""
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

