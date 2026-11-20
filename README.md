# 📊 Customer Churn: Data Quality Audit & Exploratory Analysis (DQE)

[![Track: Data Analytics](https://img.shields.io/badge/Track-Data_Analytics-blue.svg)](https://edvyro.in)
[![Level: Portfolio Ready](https://img.shields.io/badge/Level-Portfolio_Ready-brightgreen.svg)]()
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)]()
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458.svg?logo=pandas&logoColor=white)]()
[![Architecture: Modular Package](https://img.shields.io/badge/Architecture-Modular_Package-orange.svg)]()
[![Status: Completed & Validated](https://img.shields.io/badge/Audit_Status-PASSED-success.svg)]()

> **Project Track:** Data Analytics Portfolio Series  
> **Role:** Senior Data Analyst  
> **Decision Question:** *Which customer groups should a fictional subscription team contact first to reduce avoidable churn?*

---

## 📁 Project Architecture & Folder Structure

```text
DQE/
├── data/
│   ├── raw/
│   │   └── customer_churn_sample.csv       # Raw synthetic input dataset
│   └── processed/
│       └── customer_churn_cleaned.csv      # Standardized, validated production dataset
├── notebooks/
│   └── customer_churn_dqe.ipynb            # Portfolio Jupyter Notebook with analysis & plots
├── reports/
│   ├── DATA_QUALITY_REPORT.md              # Comprehensive audit & validation report
│   └── figures/                            # Publication-quality (300 DPI) visual charts
│       ├── 01_churn_by_contract_type.png
│       ├── 02_support_tickets_vs_churn.png
│       ├── 03_tenure_distribution_by_churn.png
│       ├── 04_monthly_charge_vs_tenure_scatter.png
│       └── 05_eda_summary_dashboard.png
├── src/
│   ├── __init__.py
│   ├── config.py                           # Paths, plot styling, color palettes, mappings
│   ├── profiler.py                         # Step 1: Profiling, missingness, impossible values, outliers
│   ├── cleaner.py                          # Step 2: Schema standardization & defensive assertions
│   ├── visualizer.py                       # Step 3: Seaborn/Matplotlib visualization generation
│   └── main.py                             # Pipeline orchestrator
├── run_pipeline.py                         # Root execution script
├── requirements.txt                        # Project dependencies
└── README.md                               # Project documentation & executive briefing
```

---

## 🎯 Executive Summary & Decision Question

> **Decision Question:**  
> **Which customer groups should a fictional subscription team contact first to reduce avoidable churn?**

### Prioritized Outreach Action Plan
1. **Tier 1 (Immediate Crisis Intervention):** Contact **Month-to-Month customers with $\ge 3$ support tickets in the prior 90 days**. In this dataset, this segment exhibits a **100% historical churn rate** (7 out of 7 churned accounts).
2. **Tier 2 (Proactive Onboarding Safeguard):** Intervene with **early-tenure accounts ($< 6$ months)** on Basic or Pro tiers upon submission of their **2nd support ticket**, before ticket counts reach the 3-ticket tipping point.
3. **Tier 3 (Commercial Retention Campaign):** Target **Month-to-Month accounts at the 5-month and 11-month tenure milestones** with incentives (e.g., 10–15% annual discounts) to migrate to Annual or Multi-Year commitments, eliminating monthly cancellation exposure.

---

## 🔍 Step 1: Data Profiling & Quality Audit

- **Dimensionality:** 15 rows $\times$ 11 columns in `customer_churn_sample.csv`.
- **Identifier Check:** `CustomerID` is 100% unique (15 distinct values, 0 duplicate keys).
- **Exact Duplicate Rows:** 0 duplicate records detected.
- **Completeness:** 0 missing/null cells across all columns (100% complete).
- **Domain Constraints:** All tenure values $\ge 2$ months; all charges $> 0$; $\text{TotalCharges} = \text{TenureMonths} \times \text{MonthlyCharges}$ holds exactly for all records.
- **Outlier Screening:** 0 records exceeded $1.5 \times \text{IQR}$ fences or $|Z| > 3\sigma$ bounds across all numerical metrics.

---

## 🛠️ Step 2: Reproducible Cleaning Log

| Issue Found | Affected Column / Rows | Action Taken | Rationale |
| :--- | :--- | :--- | :--- |
| Non-standard Header Casing | All 11 columns | Standardized to lowercase `snake_case` (e.g., `MonthlyCharges` $\to$ `monthly_charge`, `Churn` $\to$ `churned`) | Complies with database engineering best practices and project schema requirements. |
| Potential Trailing Whitespaces | Text features (`contract_type`, `subscription_type`, etc.) | Applied `.str.strip()` across all string columns | Eliminates potential join/filter mismatches. |
| Verbose Contract Labels | `contract_type` ('Month-to-Month', 'One Year', 'Two Year') | Created `contract_type_standardized` (`Monthly`, `Annual`, `Two-Year`) | Enhances readability on executive dashboards. |
| Categorical-only Churn Column | `churned` ('Yes', 'No') | Engineered `is_churned` binary indicator (`1` / `0`) | Enables direct mathematical calculation of churn rates. |
| Outliers / Inconvenient Values | None detected ($N=0$) | Retained all 15 customer rows | Empirically grounded observations must not be silently removed. |

### Post-Cleaning Automated Assertions
```python
assert df_clean.duplicated().sum() == 0, "Duplicate rows detected!"
assert df_clean["customer_id"].is_unique, "Customer ID primary key violation!"
assert df_clean.isnull().sum().sum() == 0, "Missing values detected!"
assert (df_clean["tenure_months"] >= 0).all(), "Negative tenure detected!"
assert (df_clean["monthly_charge"] > 0).all(), "Non-positive monthly charges detected!"
assert (df_clean["support_tickets"] >= 0).all(), "Negative support tickets detected!"
assert set(df_clean["churned"].unique()).issubset({"Yes", "No"}), "Invalid churn values!"
assert set(df_clean["contract_type_standardized"].unique()).issubset({"Monthly", "Annual", "Two-Year"}), "Invalid contract types!"
```

---

## 📈 Step 3: Summary Statistics & Exploratory Distributions

### Key Comparative Metrics: Retained ($N=8$) vs. Churned ($N=7$)

| Feature | Retained (Mean $\pm$ Std) | Retained (Median) | Churned (Mean $\pm$ Std) | Churned (Median) | Key Analytical Distinction |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tenure (Months)** | $29.12 \pm 11.48$ | 27.00 | $7.00 \pm 4.58$ | 6.00 | Churn is heavily front-loaded in early tenure ($< 15$ months). |
| **Support Tickets (90d)**| $1.00 \pm 0.76$ | 1.00 | $4.29 \pm 1.11$ | 4.00 | Churned customers filed $>4\times$ more support tickets. |
| **Monthly Charge (INR)** | $107.49 \pm 46.83$ | 114.99 | $58.56 \pm 14.64$ | 49.99 | Retained accounts skew toward higher tiers (Pro/Enterprise). |
| **Customer Age (Years)** | $46.62 \pm 9.40$ | 47.00 | $30.43 \pm 6.05$ | 29.00 | Churned subscribers in this sample were younger on average. |

### Methodological Guardrail: Correlation $\neq$ Causation
> [!IMPORTANT]
> - Filing $\ge 3$ support tickets is a **symptom of friction**, not the root cause of churn. Contacting support does not make a customer leave; unresolved technical or product dissatisfaction does.
> - Month-to-month contracts **lower switching friction**, allowing unhappy users to exit immediately without financial penalty.

---

## 📊 Visualizations

All 5 publication-ready charts are available in `reports/figures/`:
- `reports/figures/01_churn_by_contract_type.png`
- `reports/figures/02_support_tickets_vs_churn.png`
- `reports/figures/03_tenure_distribution_by_churn.png`
- `reports/figures/04_monthly_charge_vs_tenure_scatter.png`
- `reports/figures/05_eda_summary_dashboard.png`

---

## 🚀 How to Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run the Complete Modular Pipeline
```bash
python run_pipeline.py
```
Or execute directly from `src/`:
```bash
python -m src.main
```
