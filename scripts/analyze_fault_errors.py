
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from transformer_pm.dga import DGASample, engineer_features


DATASET = Path("data/processed/dga_589_ml.csv")
GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
RATIOS = [
    "RATIO_CH4_H2",
    "RATIO_C2H2_C2H4",
    "RATIO_C2H4_C2H6",
]
FEATURES = GASES + RATIOS
CLASSES = ["PD", "D1", "D2", "T1", "T2", "T3"]
RANDOM_STATE = 42


def transform_features(data):
    records = []

    for row in data.itertuples(index=False, name=None):
        sample = DGASample(*[float(value) for value in row])
        records.append(engineer_features(sample))

    return pd.DataFrame(records, index=data.index)[FEATURES]


def main():
    df = pd.read_csv(DATASET)
    X = df[GASES]
    y = df["fault_class"]

    # Exclude the original 20% holdout from development.
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_engineered = transform_features(X_train)

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    predictions = cross_val_predict(
        model,
        X_engineered,
        y_train,
        cv=cv,
        n_jobs=1,
    )

    print("=" * 65)
    print("OUT-OF-FOLD DGA FAULT ERROR ANALYSIS")
    print("=" * 65)

    print(f"Training observations: {len(y_train)}")
    print(
        "Pooled out-of-fold macro-F1:",
        round(f1_score(y_train, predictions, average="macro"), 4),
    )

    print("\nCLASSIFICATION REPORT")
    print(
        classification_report(
            y_train,
            predictions,
            labels=CLASSES,
            digits=4,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        y_train,
        predictions,
        labels=CLASSES,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES,
    )

    print("\nCONFUSION MATRIX")
    print("Rows = actual faults; columns = predicted faults")
    print(matrix_df.to_string())

    print("\nTOP FAULT CONFUSIONS")

    confusions = []
    for i, actual in enumerate(CLASSES):
        for j, predicted in enumerate(CLASSES):
            if i != j and matrix[i, j] > 0:
                confusions.append(
                    (actual, predicted, int(matrix[i, j]))
                )

    for actual, predicted, count in sorted(
        confusions,
        key=lambda item: item[2],
        reverse=True,
    )[:10]:
        print(f"{actual} -> {predicted}: {count}")

    # Save a machine-readable confusion matrix.
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    matrix_df.to_csv(output_dir / "dga_confusion_matrix.csv")

    print("\nSaved results/dga_confusion_matrix.csv")


if __name__ == "__main__":
    main()

