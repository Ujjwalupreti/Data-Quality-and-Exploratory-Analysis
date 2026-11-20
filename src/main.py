"""
Main entry point for DQE Pipeline execution.
"""
import os
import pandas as pd
from .config import (
    RAW_DATA_PATH, CLEANED_DATA_PATH, ROOT_CLEANED_DATA_PATH,
    init_visual_style
)
from .profiler import (
    audit_structure_and_completeness, audit_duplicates_and_keys,
    audit_domain_constraints, audit_outliers
)
from .cleaner import clean_dataset, validate_cleaned_dataset
from .visualizer import (
    plot_churn_by_contract, plot_support_tickets_vs_churn,
    plot_tenure_distribution, plot_monthly_charge_vs_tenure,
    plot_dashboard_panel
)

def run_dqe_pipeline():
    print("=" * 80)
    print("RUNNING MODULAR DQE PIPELINE")
    print("=" * 80)
    
    # Check raw data existence
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw dataset not found at: {RAW_DATA_PATH}")
        
    df_raw = pd.read_csv(RAW_DATA_PATH)
    print(f"\n[Step 1: Profiling] Loaded raw dataset ({len(df_raw)} rows x {len(df_raw.columns)} cols)")
    
    # Step 1: Profiling
    profile = audit_structure_and_completeness(df_raw)
    dupe_audit = audit_duplicates_and_keys(df_raw)
    domain_audit = audit_domain_constraints(df_raw)
    outliers = audit_outliers(df_raw)
    
    print("\n--- Completeness ---")
    print(profile.to_string(index=False))
    print(f"\n--- Primary Key & Duplicates ---")
    print(f"Duplicates: {dupe_audit['exact_duplicates']}, Unique PKs: {dupe_audit['unique_primary_keys']}")
    print(f"\n--- Domain Constraints ---")
    for k, v in domain_audit.items():
        print(f"{k}: {'PASS (0)' if v == 0 else f'FAIL ({v})'}")
    print(f"\n--- Outlier Screening ---")
    print(outliers[["Feature", "Min", "Median", "Max", "Lower_Bound", "Upper_Bound", "IQR_Outliers_Count", "Z_Score_Outliers_Count"]].to_string(index=False))
    
    # Step 2: Cleaning & Validation
    print("\n" + "=" * 50)
    print("Step 2: Cleaning & Validation Pipeline")
    print("=" * 50)
    df_clean = clean_dataset(df_raw)
    validate_cleaned_dataset(df_clean)
    print(">>> All automated defensive assertions passed successfully!")
    
    # Save cleaned dataset to both processed folder and project root
    export_cols = [
        "customer_id", "gender", "age", "tenure_months", "subscription_type",
        "monthly_charge", "total_charges", "contract_type", "contract_type_standardized",
        "support_tickets", "payment_method", "churned", "is_churned"
    ]
    df_clean[export_cols].to_csv(CLEANED_DATA_PATH, index=False)
    df_clean[export_cols].to_csv(ROOT_CLEANED_DATA_PATH, index=False)
    print(f"Cleaned dataset exported to:\n- {CLEANED_DATA_PATH}\n- {ROOT_CLEANED_DATA_PATH}")
    
    # Step 3: Visualization Generation
    print("\n" + "=" * 50)
    print("Step 3: Rendering Visualizations")
    print("=" * 50)
    init_visual_style()
    plot_churn_by_contract(df_clean)
    plot_support_tickets_vs_churn(df_clean)
    plot_tenure_distribution(df_clean)
    plot_monthly_charge_vs_tenure(df_clean)
    plot_dashboard_panel(df_clean)
    print(">>> All 5 high-resolution figures generated and saved in reports/figures/!")
    
    print("\n" + "=" * 80)
    print("DQE PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    run_dqe_pipeline()

