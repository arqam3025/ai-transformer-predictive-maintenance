
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
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
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATASET)

    X = df[GASES]
    y = df["fault_class"]

    # Exclude the original holdout.
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    records = []
    for row in X_train.itertuples(index=False, name=None):
        sample = DGASample(*[float(value) for value in row])
        records.append(engineer_features(sample))

    X_engineered = pd.DataFrame(
        records,
        index=X_train.index,
    )[FEATURES]

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=5,
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    scores = cross_validate(
        model,
        X_engineered,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "macro_f1": "f1_macro",
        },
        n_jobs=1,
    )

    print("=" * 65)
    print("REPEATED CROSS-VALIDATION — 5 FOLDS × 5 REPEATS")
    print("=" * 65)

    for metric in ["accuracy", "balanced_accuracy", "macro_f1"]:
        values = scores[f"test_{metric}"]

        print(f"\n{metric}")
        print(f"Mean: {np.mean(values):.4f}")
        print(f"Standard deviation: {np.std(values):.4f}")
        print(f"Minimum: {np.min(values):.4f}")
        print(f"Maximum: {np.max(values):.4f}")

    results = pd.DataFrame({
        "accuracy": scores["test_accuracy"],
        "balanced_accuracy": scores["test_balanced_accuracy"],
        "macro_f1": scores["test_macro_f1"],
    })

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    results.to_csv(
        output_dir / "dga_repeated_cv_results.csv",
        index=False,
    )

    print("\nSaved results/dga_repeated_cv_results.csv")


if __name__ == "__main__":
    main()
