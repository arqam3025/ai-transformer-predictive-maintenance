"""Prepare the transformer DGA dataset for machine-learning experiments.

Cleaning policy:
1. Preserve the original source workbook unchanged.
2. Remove exact duplicate rows.
3. Detect identical five-gas signatures associated with different labels.
4. Remove all records belonging to those contradictory signatures.
5. Translate source labels into standard project class codes.
6. Save a reproducible machine-readable training dataset.

No gas measurements are altered or imputed.
"""

from pathlib import Path

import pandas as pd


SOURCE = Path("data/raw/dga_589_training_source.xlsx")
OUTPUT = Path("data/processed/dga_589_ml.csv")

GAS_COLUMNS = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
SOURCE_LABEL = "故障类型"


LABEL_MAP = {
    "局部放电": "PD",
    "低能放电": "D1",
    "高能放电": "D2",
    "低温过热": "T1",
    "中温过热": "T2",
    "高温过热": "T3",
}


def prepare_dataset() -> pd.DataFrame:
    """Return a cleaned, unambiguous DGA dataset."""

    df = pd.read_excel(SOURCE)

    required = GAS_COLUMNS + [SOURCE_LABEL]

    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()

    if df.isna().any().any():
        raise ValueError("Source dataset contains missing values")

    if (df[GAS_COLUMNS] < 0).any().any():
        raise ValueError("Gas measurements must be non-negative")

    unknown_labels = set(df[SOURCE_LABEL].unique()) - set(LABEL_MAP)

    if unknown_labels:
        raise ValueError(
            f"Unknown source fault labels: {sorted(unknown_labels)}"
        )

    original_rows = len(df)

    # Remove exact duplicate records first.
    df = df.drop_duplicates().copy()
    after_exact_deduplication = len(df)

    # Find five-gas signatures associated with multiple source labels.
    signature_label_counts = (
        df.groupby(GAS_COLUMNS, dropna=False)[SOURCE_LABEL]
        .nunique()
        .reset_index(name="label_count")
    )

    conflicting_signatures = signature_label_counts[
        signature_label_counts["label_count"] > 1
    ][GAS_COLUMNS]

    if not conflicting_signatures.empty:
        conflicts = df.merge(
            conflicting_signatures.assign(_conflict=True),
            on=GAS_COLUMNS,
            how="left",
        )

        df = conflicts[conflicts["_conflict"].isna()].drop(
            columns="_conflict"
        )

    df = df.copy()

    # Preserve the source-language diagnosis for provenance.
    df = df.rename(columns={SOURCE_LABEL: "source_fault_label"})

    # Add standardized ML target.
    df["fault_class"] = df["source_fault_label"].map(LABEL_MAP)

    if df["fault_class"].isna().any():
        raise ValueError("Label translation produced missing classes")

    # Stable identifier for project use only.
    df.insert(
        0,
        "sample_id",
        [f"DGA-{number:04d}" for number in range(1, len(df) + 1)],
    )

    # Final deterministic order.
    df = df[
        [
            "sample_id",
            *GAS_COLUMNS,
            "source_fault_label",
            "fault_class",
        ]
    ].reset_index(drop=True)

    print(f"Original rows: {original_rows}")
    print(
        "Rows after exact duplicate removal: "
        f"{after_exact_deduplication}"
    )
    print(
        "Contradictory DGA signatures removed: "
        f"{len(conflicting_signatures)}"
    )
    print(f"Final ML observations: {len(df)}")

    return df


def main() -> None:
    df = prepare_dataset()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print(f"\nCreated: {OUTPUT}")

    print("\nFinal class distribution:")
    print(df["fault_class"].value_counts().sort_index().to_string())

    print("\nUnique gas signatures:")
    print(df[GAS_COLUMNS].drop_duplicates().shape[0])

    remaining_conflicts = (
        df.groupby(GAS_COLUMNS)["fault_class"].nunique() > 1
    ).sum()

    print(f"Remaining contradictory signatures: {remaining_conflicts}")


if __name__ == "__main__":
    main()