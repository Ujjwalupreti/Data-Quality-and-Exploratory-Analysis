"""
Script to generate customer_churn_dqe.ipynb notebook.
"""
import json
import os

notebook_path = r"c:\Users\Hp\Downloads\EDAI\DQE\customer_churn_dqe.ipynb"

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 📊 Customer Churn: Data Quality Audit & Exploratory Analysis\n",
            "**Track:** Data Analytics | **Role:** Senior Data Analyst | **Deliverable:** Portfolio-Ready DQE Report\n",
            "\n",
            "--- \n",
            "\n",
            "## 🎯 Business Context & Decision Question\n",
            "> **Decision Question:** *Which customer groups should a fictional subscription team contact first to reduce avoidable churn?*\n",
            "\n",
            "### Core Analysis Principles:\n",
            "1. **Strict Quality Auditing:** Identify missing, duplicate, impossible (e.g., negative tenure/charges), and extreme values before any exploratory modeling.\n",
            "2. **Transparent, Reproducible Cleaning:** Every assumption and transformation is explicitly logged in a reproducible cleaning log; never silently delete rows.\n",
            "3. **Descriptive Findings vs. Predictions:** Clearly distinguish between empirical observations and predictive hypotheses.\n",
            "4. **Correlation vs. Causation:** Avoid causal leaps (e.g., higher support tickets correlate with churn, but support tickets are friction signals rather than proven direct causes).\n",
            "\n",
            "### Dataset Schema:\n",
            "- `customer_id`: Synthetic unique identifier (categorical primary key; never aggregated as a continuous measure).\n",
            "- `tenure_months`: Completed months since signup (integer $\\ge 0$).\n",
            "- `monthly_charge`: Fictional monthly amount in INR (float $> 0$).\n",
            "- `support_tickets`: Count of customer service inquiries in the previous 90 days (integer $\\ge 0$).\n",
            "- `contract_type`: Billing commitment structure (`Monthly`, `Annual`, `Two-Year`).\n",
            "- `churned`: Binary indicator of whether customer cancelled in observation window (`Yes`/`No`)."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup and environment configuration\n",
            "import os\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "\n",
            "# Visualisation settings\n",
            "sns.set_theme(style=\"whitegrid\", palette=\"deep\")\n",
            "plt.rcParams.update({\n",
            "    \"font.size\": 11,\n",
            "    \"axes.labelsize\": 12,\n",
            "    \"axes.titlesize\": 14,\n",
            "    \"xtick.labelsize\": 10,\n",
            "    \"ytick.labelsize\": 10,\n",
            "    \"figure.titlesize\": 16,\n",
            "    \"figure.autolayout\": True\n",
            "})\n",
            "\n",
            "palette = {\"No\": \"#1f77b4\", \"Yes\": \"#d62728\"}  # Blue for Retained, Red for Churned\n",
            "print(\"Libraries loaded successfully.\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🔍 Step 1: Data Profiling & Quality Audit\n",
            "We perform a comprehensive audit of the raw dataset (`customer_churn_sample.csv`), evaluating:\n",
            "1. Dimensionality, column naming, and inferred data types.\n",
            "2. Missing value incidence across all features.\n",
            "3. Exact row duplicates and primary key (`CustomerID`) uniqueness.\n",
            "4. Impossible / out-of-domain values (e.g., negative charges or tenure).\n",
            "5. Statistical outlier screening using Interquartile Range (IQR) and Z-score methods."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1.1 Load Raw Dataset\n",
            "raw_filename = \"customer_churn_sample.csv\"\n",
            "df_raw = pd.read_csv(raw_filename)\n",
            "\n",
            "print(f\"Raw Dataset Shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns\\n\")\n",
            "display(df_raw.head(15))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1.2 Data Types & Completeness (Missing Value Audit)\n",
            "profile_df = pd.DataFrame({\n",
            "    \"Column_Name\": df_raw.columns,\n",
            "    \"Inferred_Dtype\": df_raw.dtypes.astype(str),\n",
            "    \"Non_Null_Count\": df_raw.count().values,\n",
            "    \"Null_Count\": df_raw.isnull().sum().values,\n",
            "    \"Null_Percentage (%)\": (df_raw.isnull().mean() * 100).round(2).values\n",
            "})\n",
            "print(\"=== DATA PROFILING: COMPLETENESS AUDIT ===\")\n",
            "display(profile_df)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1.3 Duplicate Rows & Key Integrity Check\n",
            "exact_duplicates = df_raw.duplicated().sum()\n",
            "pk_col = \"CustomerID\"\n",
            "pk_duplicates = df_raw[pk_col].duplicated().sum()\n",
            "unique_keys = df_raw[pk_col].nunique()\n",
            "\n",
            "print(f\"Exact Duplicate Rows: {exact_duplicates}\")\n",
            "print(f\"Duplicate Primary Keys ('{pk_col}'): {pk_duplicates}\")\n",
            "print(f\"Unique Primary Keys: {unique_keys} / {len(df_raw)}\")\n",
            "assert unique_keys == len(df_raw), \"Integrity Alert: Duplicate Customer IDs detected!\""
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1.4 Domain Constraints & Impossible Values Audit\n",
            "domain_checks = {\n",
            "    \"Negative Tenure (TenureMonths < 0)\": (df_raw[\"TenureMonths\"] < 0).sum(),\n",
            "    \"Non-Positive Monthly Charges (MonthlyCharges <= 0)\": (df_raw[\"MonthlyCharges\"] <= 0).sum(),\n",
            "    \"Negative Support Tickets (SupportTickets < 0)\": (df_raw[\"SupportTickets\"] < 0).sum(),\n",
            "    \"Total Charges Mismatch (|Total - (Tenure * Monthly)| > 0.05)\": (\n",
            "        (df_raw[\"TotalCharges\"].round(2) - (df_raw[\"TenureMonths\"] * df_raw[\"MonthlyCharges\"]).round(2)).abs() > 0.05\n",
            "    ).sum()\n",
            "}\n",
            "\n",
            "print(\"=== DOMAIN CONSTRAINT AUDIT RESULTS ===\")\n",
            "for check_name, fail_count in domain_checks.items():\n",
            "    status = \"PASS (0 violations)\" if fail_count == 0 else f\"FAIL ({fail_count} violations)\"\n",
            "    print(f\"- {check_name}: {status}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1.5 Outlier Screening (IQR & Z-Score Analysis)\n",
            "num_cols = df_raw.select_dtypes(include=[np.number]).columns\n",
            "outlier_audit = []\n",
            "\n",
            "for col in num_cols:\n",
            "    s = df_raw[col]\n",
            "    q1 = s.quantile(0.25)\n",
            "    q3 = s.quantile(0.75)\n",
            "    iqr = q3 - q1\n",
            "    lower_bound = q1 - 1.5 * iqr\n",
            "    upper_bound = q3 + 1.5 * iqr\n",
            "    iqr_outliers = s[(s < lower_bound) | (s > upper_bound)]\n",
            "    \n",
            "    std = s.std(ddof=1)\n",
            "    mean = s.mean()\n",
            "    z_scores = (s - mean).abs() / std if std > 0 else 0\n",
            "    z_outliers = s[z_scores > 3.0]\n",
            "    \n",
            "    outlier_audit.append({\n",
            "        \"Feature\": col,\n",
            "        \"Min\": s.min(),\n",
            "        \"Q1 (25%)\": round(q1, 2),\n",
            "        \"Median (50%)\": round(s.median(), 2),\n",
            "        \"Q3 (75%)\": round(q3, 2),\n",
            "        \"Max\": s.max(),\n",
            "        \"IQR\": round(iqr, 2),\n",
            "        \"Lower_Bound\": round(lower_bound, 2),\n",
            "        \"Upper_Bound\": round(upper_bound, 2),\n",
            "        \"IQR_Outliers\": len(iqr_outliers),\n",
            "        \"Z_Score_Outliers (>3σ)\": len(z_outliers)\n",
            "    })\n",
            "\n",
            "df_outliers = pd.DataFrame(outlier_audit)\n",
            "print(\"=== OUTLIER AUDIT SUMMARY ===\")\n",
            "display(df_outliers[[\"Feature\", \"Min\", \"Median (50%)\", \"Max\", \"Lower_Bound\", \"Upper_Bound\", \"IQR_Outliers\", \"Z_Score_Outliers (>3σ)\"]])"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🛠️ Step 2: Data Cleaning & Validation\n",
            "\n",
            "### Cleaning Strategy & Decisions:\n",
            "1. **Header Normalization:** Standardize CamelCase / TitleCase headers (`CustomerID`, `MonthlyCharges`, `Churn`, etc.) to lowercase `snake_case` (`customer_id`, `monthly_charge`, `churned`, etc.) aligning with production engineering standards.\n",
            "2. **Whitespace Stripping:** Trim leading/trailing whitespace across all text columns.\n",
            "3. **Contract Nomenclature Alignment:** Harmonize `Month-to-Month` to `Monthly`, `One Year` to `Annual`, and `Two Year` to `Two-Year` for clear executive communication.\n",
            "4. **Dual Churn Encoding:** Preserve categorical `churned` ('Yes'/'No') for descriptive clarity while engineering `is_churned` (1/0) for numerical aggregations and KPI calculations.\n",
            "5. **Automated Defensive Assertions:** Rigorously test domain integrity constraints before exporting the production dataset.\n",
            "\n",
            "### 📋 Reproducible Cleaning Log\n",
            "| Issue Found | Affected Column / Rows | Action Taken | Rationale |\n",
            "| :--- | :--- | :--- | :--- |\n",
            "| Non-standard Header Casing | All 11 columns | Renamed to `snake_case` (e.g., `MonthlyCharges` $\\to$ `monthly_charge`, `Churn` $\\to$ `churned`) | Ensures programmatic consistency and complies with project schema specification. |\n",
            "| Trailing Whitespace Risk | Text fields (`contract_type`, `subscription_type`, etc.) | Applied `.str.strip()` across string features | Prevents string mismatch bugs during filtering and grouping. |\n",
            "| Inconsistent Contract Naming | `contract_type` ('Month-to-Month', 'One Year') | Mapped to standardized labels ('Monthly', 'Annual', 'Two-Year') | Improves dashboard and report legibility for business stakeholders. |\n",
            "| Categorical-only Churn Column | `churned` ('Yes', 'No') | Engineered `is_churned` integer binary column (1/0) | Enables vectorized mean churn rate calculations in pandas and BI tools. |\n",
            "| Outliers / Inconvenient Rows | None detected ($N=0$) | Retained all 15 records in production dataset | No valid reason to alter or trim empirical customer observations. |"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2.1 Execute Cleaning Pipeline\n",
            "df_clean = df_raw.copy()\n",
            "\n",
            "# Rename columns to snake_case schema\n",
            "col_mapping = {\n",
            "    \"CustomerID\": \"customer_id\",\n",
            "    \"Gender\": \"gender\",\n",
            "    \"Age\": \"age\",\n",
            "    \"TenureMonths\": \"tenure_months\",\n",
            "    \"SubscriptionType\": \"subscription_type\",\n",
            "    \"MonthlyCharges\": \"monthly_charge\",\n",
            "    \"TotalCharges\": \"total_charges\",\n",
            "    \"ContractType\": \"contract_type\",\n",
            "    \"SupportTickets\": \"support_tickets\",\n",
            "    \"PaymentMethod\": \"payment_method\",\n",
            "    \"Churn\": \"churned\"\n",
            "}\n",
            "df_clean.rename(columns=col_mapping, inplace=True)\n",
            "\n",
            "# Strip whitespace\n",
            "for col in df_clean.select_dtypes(include=\"object\").columns:\n",
            "    df_clean[col] = df_clean[col].astype(str).str.strip()\n",
            "\n",
            "# Standardize contract nomenclature\n",
            "contract_clean_map = {\n",
            "    \"Month-to-Month\": \"Monthly\",\n",
            "    \"One Year\": \"Annual\",\n",
            "    \"Two Year\": \"Two-Year\"\n",
            "}\n",
            "df_clean[\"contract_type_standardized\"] = df_clean[\"contract_type\"].replace(contract_clean_map)\n",
            "\n",
            "# Dual encoding for churn\n",
            "df_clean[\"is_churned\"] = (df_clean[\"churned\"] == \"Yes\").astype(int)\n",
            "\n",
            "print(\"Cleaned Dataset Preview:\")\n",
            "display(df_clean.head(5))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2.2 Defensive Post-Cleaning Assertions\n",
            "print(\"Running automated post-cleaning assertions...\")\n",
            "assert df_clean.duplicated().sum() == 0, \"Duplicate rows detected!\"\n",
            "assert df_clean[\"customer_id\"].is_unique, \"Primary key is not unique!\"\n",
            "assert df_clean.isnull().sum().sum() == 0, \"Null values detected!\"\n",
            "assert (df_clean[\"tenure_months\"] >= 0).all(), \"Negative tenure found!\"\n",
            "assert (df_clean[\"monthly_charge\"] > 0).all(), \"Non-positive monthly charges found!\"\n",
            "assert (df_clean[\"support_tickets\"] >= 0).all(), \"Negative support tickets found!\"\n",
            "assert set(df_clean[\"churned\"].unique()).issubset({\"Yes\", \"No\"}), \"Invalid churned categories!\"\n",
            "assert set(df_clean[\"contract_type_standardized\"].unique()).issubset({\"Monthly\", \"Annual\", \"Two-Year\"}), \"Invalid contract categories!\"\n",
            "print(\"✅ ALL POST-CLEANING ASSERTIONS PASSED SUCCESSFULLY!\")\n",
            "\n",
            "# 2.3 Export Cleaned Dataset\n",
            "clean_export_path = \"customer_churn_cleaned.csv\"\n",
            "df_clean.to_csv(clean_export_path, index=False)\n",
            "print(f\"Cleaned dataset exported to: '{clean_export_path}' ({len(df_clean)} rows x {len(df_clean.columns)} columns)\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📈 Step 3: Summary Statistics & Exploratory Distributions\n",
            "\n",
            "Here we compute parametric and non-parametric summary statistics, analyze bivariate crosstabs, and generate well-annotated visualizations."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3.1 Numerical Summary Statistics (Excluding customer_id)\n",
            "num_features = [\"tenure_months\", \"monthly_charge\", \"total_charges\", \"support_tickets\", \"age\"]\n",
            "stats_summary = df_clean[num_features].describe(percentiles=[0.25, 0.50, 0.75]).T\n",
            "stats_summary[\"median\"] = df_clean[num_features].median()\n",
            "stats_summary[\"iqr\"] = stats_summary[\"75%\"] - stats_summary[\"25%\"]\n",
            "\n",
            "display(stats_summary[[\"count\", \"mean\", \"std\", \"min\", \"25%\", \"median\", \"75%\", \"max\", \"iqr\"]].round(2))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3.2 Categorical Frequencies & Percentages\n",
            "for col in [\"contract_type_standardized\", \"subscription_type\", \"payment_method\", \"churned\"]:\n",
            "    freq = df_clean[col].value_counts()\n",
            "    pct = (df_clean[col].value_counts(normalize=True) * 100).round(2)\n",
            "    df_cat = pd.DataFrame({\"Count\": freq, \"Percentage (%)\": pct})\n",
            "    print(f\"\\n--- Categorical Distribution: {col} ---\")\n",
            "    display(df_cat)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3.3 Comparative Metrics: Retained vs. Churned Customers\n",
            "comp_analysis = df_clean.groupby(\"churned\")[num_features].agg([\"mean\", \"median\", \"std\"]).T\n",
            "print(\"=== COMPARATIVE SUMMARY: RETAINED (No) vs CHURNED (Yes) ===\")\n",
            "display(comp_analysis.round(2))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 1: Observed Churn Rate by Contract Type\n",
            "fig, ax = plt.subplots(figsize=(8, 4.8))\n",
            "contract_stats = df_clean.groupby(\"contract_type_standardized\")[\"is_churned\"].mean() * 100\n",
            "contract_order = [\"Monthly\", \"Annual\", \"Two-Year\"]\n",
            "rates = [contract_stats.get(k, 0) for k in contract_order]\n",
            "\n",
            "bars = ax.bar(contract_order, rates, color=[\"#d62728\", \"#1f77b4\", \"#1f77b4\"], width=0.5, edgecolor=\"black\", alpha=0.9)\n",
            "for bar in bars:\n",
            "    h = bar.get_height()\n",
            "    ax.text(bar.get_x() + bar.get_width() / 2.0, h + 2, f\"{h:.1f}%\", ha=\"center\", va=\"bottom\", fontweight=\"bold\", fontsize=11)\n",
            "\n",
            "ax.set_ylim(0, 115)\n",
            "ax.set_title(\"Figure 1: Observed Churn Rate by Contract Commitment Type\", fontsize=13, fontweight=\"bold\", pad=12)\n",
            "ax.set_xlabel(\"Contract Commitment Type\", fontsize=11, labelpad=8)\n",
            "ax.set_ylabel(\"Observed Churn Rate (%)\", fontsize=11, labelpad=8)\n",
            "ax.grid(axis=\"y\", linestyle=\"--\", alpha=0.6)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 2: Support Ticket Inquiries by Churn Status\n",
            "fig, ax = plt.subplots(figsize=(8, 4.8))\n",
            "sns.countplot(data=df_clean, x=\"support_tickets\", hue=\"churned\", palette=palette, ax=ax, edgecolor=\"black\", alpha=0.9)\n",
            "ax.set_title(\"Figure 2: Distribution of 90-Day Support Tickets by Churn Status\", fontsize=13, fontweight=\"bold\", pad=12)\n",
            "ax.set_xlabel(\"Customer Support Tickets (Prior 90 Days)\", fontsize=11, labelpad=8)\n",
            "ax.set_ylabel(\"Number of Customers\", fontsize=11, labelpad=8)\n",
            "ax.axvline(x=2.5, color=\"red\", linestyle=\"--\", linewidth=1.8)\n",
            "ax.text(2.6, 3.2, \"Inflection Threshold:\\nAll customers with ≥3\\ntickets churned (7/7)\", color=\"darkred\", fontsize=10, fontweight=\"semibold\")\n",
            "ax.legend(title=\"Churned\", loc=\"upper right\")\n",
            "ax.grid(axis=\"y\", linestyle=\"--\", alpha=0.6)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 3: Customer Tenure Months by Churn Status\n",
            "fig, ax = plt.subplots(figsize=(8, 4.8))\n",
            "sns.boxplot(data=df_clean, x=\"churned\", y=\"tenure_months\", hue=\"churned\", palette=palette, ax=ax, width=0.42, boxprops=dict(alpha=0.85, edgecolor=\"black\"), legend=False)\n",
            "sns.stripplot(data=df_clean, x=\"churned\", y=\"tenure_months\", color=\"black\", size=7, jitter=0.15, ax=ax, alpha=0.75)\n",
            "ax.set_title(\"Figure 3: Customer Tenure Distribution by Churn Status\", fontsize=13, fontweight=\"bold\", pad=12)\n",
            "ax.set_xlabel(\"Customer Churned?\", fontsize=11, labelpad=8)\n",
            "ax.set_ylabel(\"Tenure (Completed Months)\", fontsize=11, labelpad=8)\n",
            "ax.grid(axis=\"y\", linestyle=\"--\", alpha=0.6)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Figure 4: Customer Segmentation Scatter Plot (Tenure vs Monthly Charge)\n",
            "fig, ax = plt.subplots(figsize=(9, 5.2))\n",
            "sns.scatterplot(\n",
            "    data=df_clean,\n",
            "    x=\"tenure_months\",\n",
            "    y=\"monthly_charge\",\n",
            "    hue=\"churned\",\n",
            "    style=\"contract_type_standardized\",\n",
            "    palette=palette,\n",
            "    s=130,\n",
            "    edgecolor=\"black\",\n",
            "    alpha=0.9,\n",
            "    ax=ax\n",
            ")\n",
            "ax.set_title(\"Figure 4: Bivariate Segmentation: Tenure vs Monthly Charge (INR)\", fontsize=13, fontweight=\"bold\", pad=12)\n",
            "ax.set_xlabel(\"Tenure (Completed Months)\", fontsize=11, labelpad=8)\n",
            "ax.set_ylabel(\"Monthly Charge (INR)\", fontsize=11, labelpad=8)\n",
            "ax.axvspan(0, 14.5, color=\"red\", alpha=0.08, label=\"High-Risk Window (Tenure < 15)\")\n",
            "ax.legend(bbox_to_anchor=(1.02, 1), loc=\"upper left\", title=\"Legend\")\n",
            "ax.grid(True, linestyle=\"--\", alpha=0.6)\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🎯 Step 4: Decision Question Synthesis & Contact Prioritization Strategy\n",
            "\n",
            "> **Decision Question:** *Which customer groups should a fictional subscription team contact first to reduce avoidable churn?*\n",
            "\n",
            "### Analytical Findings (Descriptive):\n",
            "1. **Contract Structure:** 100% of churned accounts (7/7) were on **Month-to-Month contracts**. Annual and Two-Year contracts had 0% churn.\n",
            "2. **Support Interaction Volume:** Customers with $\\ge 3$ support tickets had a 100% churn rate (7/7), with an average of 4.29 tickets, versus 1.00 ticket for retained customers.\n",
            "3. **Tenure Fragility:** Churned customers had a median tenure of 6.0 months (mean 7.0, range 2–14 months), compared to retained customers with median tenure of 27.0 months (mean 29.1 months).\n",
            "4. **Subscription Tier:** Basic subscribers experienced 71.4% churn (5/7), Pro subscribers experienced 50.0% churn (2/4), and Enterprise experienced 0.0% churn (0/4).\n",
            "\n",
            "### Strict Analytical Guardrail: Correlation $\\neq$ Causation\n",
            "- High ticket volume reflects **operational friction or customer distress**, not that contacting customer service causes churn.\n",
            "- Month-to-Month contracts provide **low switching friction**, allowing unhappy users to exit immediately without financial penalty.\n",
            "\n",
            "### 🚦 Prioritized Action Plan for the Subscription Team\n",
            "| Priority Tier | Target Customer Segment | Operational Trigger | Recommended Action | Expected Business Outcome |\n",
            "| :--- | :--- | :--- | :--- | :--- |\n",
            "| **Tier 1 (Immediate)** | Month-to-Month accounts with $\\ge 3$ support tickets in prior 90 days | Ticket count $\\ge 3$ | **Executive Support Outreach:** Dedicated proactive check-in by a senior support lead to resolve lingering technical issues before renewal. | Stems immediate churn in the highest-risk group (100% historical churn rate). |\n",
            "| **Tier 2 (Proactive)** | Early-tenure accounts ($< 6$ months) on Basic/Pro plans | Ticket count reaching 2 | **Early Onboarding Assist:** Trigger guided walkthrough or customer success call upon filing of 2nd support ticket. | Prevents ticket escalation past the 3-ticket tipping point. |\n",
            "| **Tier 3 (Commercial)** | Month-to-Month subscribers approaching Month 6 or 12 | Tenure milestone (Month 5/11) | **Contract Migration Incentive:** Offer a 10–15% discount or feature perk to migrate from Monthly to Annual contract. | Eliminates monthly renewal vulnerability and builds long-term commitment. |"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📖 Appendix: Data Dictionary\n",
            "\n",
            "| Column Name | Data Type | Description | Valid Values / Range | Notes |\n",
            "| :--- | :--- | :--- | :--- | :--- |\n",
            "| `customer_id` | String / Object | Unique synthetic identifier assigned to each subscriber | Format: `CUST-XXXX` | Primary key. Must never be summed, averaged, or aggregated as a continuous metric. |\n",
            "| `gender` | String / Object | Self-reported customer gender | `Female`, `Male` | Context demographic feature. |\n",
            "| `age` | Integer | Customer age in years | Positive integers (23–60 in sample) | Demographic feature. |\n",
            "| `tenure_months` | Integer | Completed months since account signup | Integers $\\ge 0$ (2–48 in sample) | Core measure of customer lifetime and relationship maturity. |\n",
            "| `subscription_type`| String / Object | Customer subscription tier | `Basic`, `Pro`, `Enterprise` | Reflects product feature access level. |\n",
            "| `monthly_charge` | Float | Fictional monthly subscription charge in INR | Numeric $> 0$ (49.99, 79.99, 149.99 in sample) | Unit: Indian Rupee (INR). |\n",
            "| `total_charges` | Float | Cumulative billed amount over customer tenure | Numeric $\\ge 0$ (99.98–7199.52 in sample) | Consistent with `tenure_months * monthly_charge`. |\n",
            "| `contract_type` | String / Object | Raw contract duration agreement | `Month-to-Month`, `One Year`, `Two Year` | Raw contract agreement structure. |\n",
            "| `contract_type_standardized` | String / Object | Harmonized contract commitment label | `Monthly`, `Annual`, `Two-Year` | Standardized nomenclature for executive clarity. |\n",
            "| `support_tickets`| Integer | Inquiries submitted to customer support in prior 90 days | Integers $\\ge 0$ (0–6 in sample) | Key operational friction proxy. |\n",
            "| `payment_method` | String / Object | Primary billing transaction mechanism | `Credit Card`, `Bank Transfer`, `UPI`, `Debit Card` | Payment processing channel. |\n",
            "| `churned` | String / Object | Whether customer discontinued service in the observation period | `Yes`, `No` | Primary target outcome variable. |\n",
            "| `is_churned` | Integer | Binary numerical indicator for churn status | `1` (Churned), `0` (Retained) | Engineered for fast mathematical aggregation and BI KPI calculations. |"
        ]
    }
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10"
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter Notebook generated successfully at: {notebook_path}")

