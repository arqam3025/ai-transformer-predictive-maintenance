
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from transformer_pm.dga import DGASample, engineer_features


DATASET = Path("data/processed/dga_589_ml.csv")
GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATASET)

    X = df[GASES]
    y = df["fault_class"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    engineered = []

    for row in X_train.itertuples(index=False, name=None):
        sample = DGASample(*[float(value) for value in row])
        engineered.append(engineer_features(sample))

    X_features = pd.DataFrame(
        engineered,
        index=X_train.index,
    )

    ratio_features = [
        "RATIO_CH4_H2",
        "RATIO_C2H2_C2H4",
        "RATIO_C2H4_C2H6",
    ]

    fraction_features = [
        f"FRACTION_{gas}" for gas in GASES
    ]

    configurations = {
        "A: Original gases": GASES,
        "B: Gases + total": GASES + ["TOTAL_GAS"],
        "C: Gases + ratios": GASES + ratio_features,
        "D: Gases + fractions": GASES + fraction_features,
        "E: All 14 features": list(X_features.columns),
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    for name, columns in configurations.items():
        model = RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        scores = cross_validate(
            model,
            X_features[columns],
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "macro_f1": "f1_macro",
                "balanced_accuracy": "balanced_accuracy",
            },
            n_jobs=1,
        )

        results.append({
            "configuration": name,
            "features": len(columns),
            "accuracy": scores["test_accuracy"].mean(),
            "macro_f1": scores["test_macro_f1"].mean(),
            "macro_f1_std": scores["test_macro_f1"].std(),
            "balanced_accuracy":
                scores["test_balanced_accuracy"].mean(),
        })

    results_df = pd.DataFrame(results).sort_values(
        "macro_f1",
        ascending=False,
    )

    print("\n" + "=" * 75)
    print("DGA FEATURE ABLATION STUDY — TRAINING DATA ONLY")
    print("=" * 75)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
