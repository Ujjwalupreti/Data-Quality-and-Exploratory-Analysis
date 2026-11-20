"""
Step 3: Visualizations & Exploratory Plotting Module.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .config import PALETTE_CHURN, FIGURES_DIR, ROOT_FIGURES_DIR

def plot_churn_by_contract(df: pd.DataFrame) -> str:
    """Figure 1: Observed Churn Rate by Contract Type."""
    fig, ax = plt.subplots(figsize=(8, 5))
    rates = df.groupby("contract_type_standardized")["is_churned"].mean() * 100
    bars = ax.bar(rates.index, rates.values, color=["#d62728" if v > 50 else "#1f77b4" for v in rates.values], width=0.55, edgecolor="black", alpha=0.9)
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
        
    ax.set_ylim(0, 115)
    ax.set_title("Descriptive Finding: Churn Rate by Contract Type", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Contract Type", fontsize=12, labelpad=8)
    ax.set_ylabel("Observed Churn Rate (%)", fontsize=12, labelpad=8)
    
    path_report = os.path.join(FIGURES_DIR, "01_churn_by_contract_type.png")
    fig.savefig(path_report, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(ROOT_FIGURES_DIR, "01_churn_by_contract_type.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path_report

def plot_support_tickets_vs_churn(df: pd.DataFrame) -> str:
    """Figure 2: Support Tickets vs Churn Status."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x="support_tickets", hue="churned", palette=PALETTE_CHURN, ax=ax, edgecolor="black", alpha=0.9)
    ax.set_title("Distribution of 90-Day Support Tickets by Churn Status", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Support Tickets (Count in Prior 90 Days)", fontsize=12, labelpad=8)
    ax.set_ylabel("Customer Count", fontsize=12, labelpad=8)
    ax.axvline(x=2.5, color="red", linestyle="--", linewidth=1.8)
    ax.text(2.6, 3.5, "Sharp Inflection:\nAll customers with >=3\ntickets churned (7/7)", color="darkred", fontsize=10, fontweight="semibold")
    ax.legend(title="Churned", loc="upper right")
    
    path_report = os.path.join(FIGURES_DIR, "02_support_tickets_vs_churn.png")
    fig.savefig(path_report, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(ROOT_FIGURES_DIR, "02_support_tickets_vs_churn.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path_report

def plot_tenure_distribution(df: pd.DataFrame) -> str:
    """Figure 3: Tenure Months Distribution by Churn Status."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="churned", y="tenure_months", hue="churned", palette=PALETTE_CHURN, ax=ax, width=0.45, boxprops=dict(alpha=0.85, edgecolor="black"), legend=False)
    sns.stripplot(data=df, x="churned", y="tenure_months", color="black", size=7, jitter=0.15, ax=ax, alpha=0.75)
    ax.set_title("Tenure Months Distribution by Churn Status", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Customer Churned?", fontsize=12, labelpad=8)
    ax.set_ylabel("Tenure (Completed Months)", fontsize=12, labelpad=8)
    
    path_report = os.path.join(FIGURES_DIR, "03_tenure_distribution_by_churn.png")
    fig.savefig(path_report, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(ROOT_FIGURES_DIR, "03_tenure_distribution_by_churn.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path_report

def plot_monthly_charge_vs_tenure(df: pd.DataFrame) -> str:
    """Figure 4: Monthly Charge vs Tenure Scatter Plot."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.scatterplot(
        data=df,
        x="tenure_months",
        y="monthly_charge",
        hue="churned",
        style="contract_type_standardized",
        palette=PALETTE_CHURN,
        s=140,
        ax=ax,
        edgecolor="black",
        alpha=0.9
    )
    ax.set_title("Customer Segmentation: Tenure vs Monthly Charge (INR)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Tenure (Completed Months)", fontsize=12, labelpad=8)
    ax.set_ylabel("Monthly Charge (INR)", fontsize=12, labelpad=8)
    ax.axvspan(0, 15, color="red", alpha=0.08, label="High-Risk Zone (Tenure < 15 mos)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    
    path_report = os.path.join(FIGURES_DIR, "04_monthly_charge_vs_tenure_scatter.png")
    fig.savefig(path_report, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(ROOT_FIGURES_DIR, "04_monthly_charge_vs_tenure_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path_report

def plot_dashboard_panel(df: pd.DataFrame) -> str:
    """Figure 5: 2x2 Portfolio Visualisation Panel."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Customer Churn Data Quality & Exploratory Analysis Dashboard", fontsize=16, fontweight="bold", y=0.99)
    
    # 1. Churn by Contract
    rates = df.groupby("contract_type_standardized")["is_churned"].mean() * 100
    bars1 = axes[0, 0].bar(rates.index, rates.values, color=["#d62728" if v > 50 else "#1f77b4" for v in rates.values], width=0.5, edgecolor="black")
    for b in bars1:
        h = b.get_height()
        axes[0, 0].text(b.get_x() + b.get_width() / 2.0, h + 2, f"{h:.0f}%", ha="center", va="bottom", fontweight="bold")
    axes[0, 0].set_ylim(0, 115)
    axes[0, 0].set_title("A: Churn Rate by Contract Type", fontweight="bold")
    axes[0, 0].set_ylabel("Churn Rate (%)")
    
    # 2. Tickets by Churn
    sns.countplot(data=df, x="support_tickets", hue="churned", palette=PALETTE_CHURN, ax=axes[0, 1], edgecolor="black", alpha=0.9)
    axes[0, 1].set_title("B: Support Tickets (Prior 90 Days) by Churn", fontweight="bold")
    axes[0, 1].axvline(x=2.5, color="red", linestyle="--", linewidth=1.5)
    axes[0, 1].legend(title="Churned", loc="upper right")
    
    # 3. Tenure Boxplot
    sns.boxplot(data=df, x="churned", y="tenure_months", hue="churned", palette=PALETTE_CHURN, ax=axes[1, 0], width=0.4, boxprops=dict(alpha=0.85, edgecolor="black"), legend=False)
    sns.stripplot(data=df, x="churned", y="tenure_months", color="black", size=6, jitter=0.15, ax=axes[1, 0], alpha=0.7)
    axes[1, 0].set_title("C: Tenure Months Distribution by Churn", fontweight="bold")
    
    # 4. Plan Churn
    sub_churn = df.groupby("subscription_type")["is_churned"].mean() * 100
    sub_order = ["Basic", "Pro", "Enterprise"]
    bars4 = axes[1, 1].bar(sub_order, [sub_churn.get(k, 0) for k in sub_order], color=["#d62728", "#ff7f0e", "#2ca02c"], width=0.5, edgecolor="black", alpha=0.9)
    for b in bars4:
        h = b.get_height()
        axes[1, 1].text(b.get_x() + b.get_width() / 2.0, h + 2, f"{h:.1f}%", ha="center", va="bottom", fontweight="bold")
    axes[1, 1].set_ylim(0, 100)
    axes[1, 1].set_title("D: Churn Rate by Subscription Tier", fontweight="bold")
    
    path_report = os.path.join(FIGURES_DIR, "05_eda_summary_dashboard.png")
    fig.savefig(path_report, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(ROOT_FIGURES_DIR, "05_eda_summary_dashboard.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path_report

