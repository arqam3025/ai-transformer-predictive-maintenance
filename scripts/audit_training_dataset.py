"""Audit the 589-sample transformer DGA training dataset.

This script performs data-quality checks before any machine-learning
training. The original Excel workbook is read-only and is never modified.
"""

from pathlib import Path

import pandas as pd


SOURCE = Path("data/raw/dga_589_training_source.xlsx")

GAS_COLUMNS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
LABEL_COLUMN = "故障类型"


def load_dataset() -> pd.DataFrame:
    """Load the original DGA workbook without modifying it."""

    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    return pd.read_excel(SOURCE)


def main() -> None:
    df = load_dataset()

    print("=" * 65)
    print("DGA TRAINING DATASET AUDIT")
    print("=" * 65)

    print(f"\nSource: {SOURCE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Column names: {df.columns.tolist()}")

    # Schema validation
    expected = GAS_COLUMNS + [LABEL_COLUMN]
    missing_columns = [column for column in expected if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Missing values
    print("\n--- Missing values ---")
    print(df[expected].isna().sum().to_string())

    # Negative gas measurements
    negative_counts = (df[GAS_COLUMNS] < 0).sum()

    print("\n--- Negative gas measurements ---")
    print(negative_counts.to_string())

    # Gas statistics
    print("\n--- Gas statistics ---")
    print(df[GAS_COLUMNS].describe().to_string())

    # Class distribution
    print("\n--- Original fault labels ---")
    class_counts = df[LABEL_COLUMN].value_counts(dropna=False)

    for label, count in class_counts.items():
        print(f"{label!r}: {count}")

    # Exact duplicate rows including label
    exact_duplicate_mask = df.duplicated(keep=False)
    exact_duplicate_rows = df[exact_duplicate_mask]

    print("\n--- Exact duplicate audit ---")
    print(f"Rows belonging to duplicated full records: {len(exact_duplicate_rows)}")
    print(
        "Number of duplicate rows beyond first occurrence: "
        f"{df.duplicated().sum()}"
    )

    # Duplicate gas signatures regardless of label
    repeated_gas_mask = df.duplicated(subset=GAS_COLUMNS, keep=False)
    repeated_gas_rows = df[repeated_gas_mask].sort_values(GAS_COLUMNS)

    print("\n--- Repeated DGA signatures ---")
    print(
        "Rows whose five-gas measurements occur more than once: "
        f"{len(repeated_gas_rows)}"
    )

    # Cross-label collisions:
    # identical five-gas measurements associated with >1 diagnosis
    label_counts_per_signature = (
        df.groupby(GAS_COLUMNS, dropna=False)[LABEL_COLUMN]
        .nunique(dropna=False)
        .reset_index(name="label_count")
    )

    conflicting_signatures = label_counts_per_signature[
        label_counts_per_signature["label_count"] > 1
    ]

    print("\n--- Cross-label collision audit ---")
    print(
        "Unique DGA signatures associated with multiple labels: "
        f"{len(conflicting_signatures)}"
    )

    if not conflicting_signatures.empty:
        conflicts = df.merge(
            conflicting_signatures[GAS_COLUMNS],
            on=GAS_COLUMNS,
            how="inner",
        ).sort_values(GAS_COLUMNS + [LABEL_COLUMN])

        print("\nConflicting records:")
        print(conflicts.to_string(index=False))

    # Unique gas signatures
    unique_signatures = df[GAS_COLUMNS].drop_duplicates()

    print("\n--- Summary ---")
    print(f"Total observations: {len(df)}")
    print(f"Unique five-gas signatures: {len(unique_signatures)}")
    print(f"Number of source labels: {df[LABEL_COLUMN].nunique()}")

    print("\nAudit complete. Source workbook was not modified.")


if __name__ == "__main__":
    main()