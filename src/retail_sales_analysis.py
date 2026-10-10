import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "superstore_sales.csv"
CLEAN_FILE = BASE_DIR / "data" / "cleaned" / "superstore_sales_cleaned.csv"
REPORT_DIR = BASE_DIR / "reports"


def load_data():
    """Load the raw retail sales dataset."""
    return pd.read_csv(RAW_FILE)


def clean_data(df):
    """Remove duplicates and prepare date and numeric columns."""
    df = df.drop_duplicates().copy()
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
    df = df.dropna(subset=["Order Date", "Sales", "Profit", "Quantity"])
    df["Month"] = df["Order Date"].dt.to_period("M")
    return df


def create_dashboard(df):
    """Create sales and profit charts."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sales_category = df.groupby("Category")["Sales"].sum()
    monthly_sales = df.groupby("Month")["Sales"].sum()
    top_subcategories = (
        df.groupby("Sub-Category")["Sales"]
        .sum()
        .nlargest(10)
    )
    profit_category = df.groupby("Category")["Profit"].sum()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    sales_category.plot(kind="bar", ax=axes[0, 0], title="Sales by Category")
    monthly_sales.plot(ax=axes[0, 1], title="Monthly Sales Trend")
    top_subcategories.plot(
        kind="bar", ax=axes[1, 0], title="Top 10 Sub-Categories by Sales"
    )
    profit_category.plot(kind="bar", ax=axes[1, 1], title="Profit by Category")

    for ax in axes.flat:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "retail_sales_dashboard.png", dpi=150)
    plt.close(fig)


def main():
    df = load_data()
    cleaned_df = clean_data(df)

    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEAN_FILE, index=False)

    create_dashboard(cleaned_df)

    print("Analysis completed successfully.")
    print(f"Original records: {len(df)}")
    print(f"Cleaned records: {len(cleaned_df)}")
    print(f"Dashboard saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
