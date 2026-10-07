"""Build the 25-transformer DGA validation dataset.

The raw measurements and fault descriptions are transcribed from the
source dataset. This script preserves the original diagnosis while adding
a conservative normalized label for later analysis.

This dataset is intended for validation/reference, not for training a
production-grade machine-learning classifier.
"""

from pathlib import Path

import pandas as pd


OUTPUT_PATH = Path("data/processed/dga_25_transformers.csv")

COLUMNS = [
    "transformer_id",
    "H2",
    "CH4",
    "C2H6",
    "C2H4",
    "C2H2",
    "fault_description",
    "fault_class",
]


RECORDS = [
    (1, 1634, 136, 7, 1, 0, "PD", "PD"),
    (
        2,
        108.9,
        2.48,
        16.6,
        16.62,
        0,
        "T1 - Unsatisfactory insulation of electrical steel sheets",
        "T1",
    ),
    (
        3,
        50,
        4.8,
        5.2,
        23,
        1.1,
        "Critical PD - Loose nut on LV winding lead with bushing stud",
        "PD",
    ),
    (4, 169, 22, 1, 0.001, 0, "Corona discharge", "OTHER_DISCHARGE"),
    (
        5,
        60,
        28.2,
        6.5,
        59.6,
        17.1,
        "High intensity spark discharge. Inspection revealed a mechanical "
        "break in the C phase",
        "OTHER_DISCHARGE",
    ),
    (
        6,
        45,
        30,
        13,
        30,
        18,
        "Sustained developed PD with overlapping on the surface of the solid insulation",
        "PD",
    ),
    (
        7,
        160,
        90,
        14,
        140,
        100,
        "Extensive traces of creeping discharges on the surface and in the "
        "thickness of the main insulation",
        "OTHER_DISCHARGE",
    ),
    (8, 2000, 210, 60, 270, 1340, "D2 - Broken switch current lead", "D2"),
    (
        9,
        30,
        800,
        170,
        10,
        0,
        "T1 - Heating of the pressing bolts by leakage currents",
        "T1",
    ),
    (
        10,
        13,
        131,
        37,
        5,
        0.2,
        "T1 - Loose pressing of the magnetic core",
        "T1",
    ),
    (
        11,
        40,
        66,
        35,
        56,
        0,
        "T2 - Increased heating of the bolt connections",
        "T2",
    ),
    (
        12,
        17.1,
        102,
        161,
        28.5,
        0.8,
        "T1 - Presence of short-circuited circuit",
        "T1",
    ),
    (
        13,
        20,
        100,
        7000,
        200,
        10,
        "T1 - Contamination of pipes and inter-tube space, clogged cooler pipes",
        "T1",
    ),
    (
        14,
        10,
        30,
        520,
        120,
        130,
        "Spark discharge accompanied by heating up to 300 C. Inspection "
        "revealed a bad contact in the lower part of the HV winding",
        "OTHER_DISCHARGE",
    ),
    (
        15,
        47.3,
        181,
        81,
        415,
        18.6,
        "T3 - Overheating of the iron due to faulty circulation of currents in the core",
        "T3",
    ),
    (
        16,
        104.5,
        144.5,
        15.6,
        178.8,
        5.6,
        "T3 - Thermal defect caused by faulty connections",
        "T3",
    ),
    (
        17,
        52,
        45,
        22,
        180,
        11,
        "Overheating caused by a short circuit on the magnetic core. Discharge traces",
        "MIXED_UNSPECIFIED",
    ),
    (
        18,
        870,
        1100,
        191,
        1907,
        191,
        "Traces of abnormal heating and discharges",
        "MIXED_UNSPECIFIED",
    ),
    (
        19,
        65.2,
        42.7,
        18.3,
        165.3,
        88.4,
        "T3 and D2 - The tensioned stud having signs of arcing discharges at the ends",
        "T3+D2",
    ),
    (
        20,
        525,
        109,
        113,
        527,
        420,
        "T3 and D2 - Arc in the yoke beam of the top of the transformer",
        "T3+D2",
    ),
    (
        21,
        120,
        79,
        50,
        390,
        8100,
        "D2 - Loosening of the nut on the LV winding bushing stud",
        "D2",
    ),
    (
        22,
        60,
        140,
        60,
        60,
        250,
        "T2 and D1 - Contact between top bracket and tank rail with traces of molten metal",
        "T2+D1",
    ),
    (
        23,
        160,
        24,
        6,
        150,
        400,
        "D2 - Breakdown of the winding insulation",
        "D2",
    ),
    (24, 940, 730, 250, 2280, 2470, "D2", "D2"),
    (
        25,
        245,
        30,
        5,
        35,
        245,
        "D2 - Electrical overlap between windings along the resistance rail was detected",
        "D2",
    ),
]


def build_dataset() -> pd.DataFrame:
    """Build and validate the structured DGA reference dataset."""

    df = pd.DataFrame(RECORDS, columns=COLUMNS)

    if len(df) != 25:
        raise ValueError(f"Expected 25 transformer records, found {len(df)}")

    if not df["transformer_id"].is_unique:
        raise ValueError("Transformer IDs must be unique")

    gas_columns = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]

    if df[gas_columns].isna().any().any():
        raise ValueError("DGA gas measurements must not contain missing values")

    if (df[gas_columns] < 0).any().any():
        raise ValueError("DGA gas measurements must be non-negative")

    if df["fault_description"].isna().any():
        raise ValueError("Fault descriptions must be preserved")

    if df["fault_class"].isna().any():
        raise ValueError("Every record requires an explicit analysis label")

    return df


def main() -> None:
    """Generate the processed validation CSV."""

    df = build_dataset()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Created: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print("\nFault-class counts:")
    print(df["fault_class"].value_counts().sort_index())


if __name__ == "__main__":
    main()