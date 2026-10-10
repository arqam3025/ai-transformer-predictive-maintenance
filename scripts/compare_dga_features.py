
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from transformer_pm.dga import DGASample, engineer_features


DATASET = Path("data/processed/dga_589_ml.csv")
GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
RANDOM_STATE = 42


def make_engineered_features(X):
    records = []

    for row in X.itertuples(index=False, name=None):
        sample = DGASample(*[float(value) for value in row])
        records.append(engineer_features(sample))

    return pd.DataFrame(records, index=X.index)


def main():
    df = pd.read_csv(DATASET)

    X = df[GASES].copy()
    y = df["fault_class"].copy()

    # Reproduce the original training split.
    from sklearn.model_selection import train_test_split

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_enhanced = make_engineered_features(X_train)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("=" * 65)
    print("DGA FEATURE ENGINEERING EXPERIMENT")
    print("=" * 65)

    print(f"Training samples: {len(X_train)}")
    print(f"Original features: {X_train.shape[1]}")
    print(f"Enhanced features: {X_enhanced.shape[1]}")

    results = []

    for name, features in [
        ("Original five gases", X_train),
        ("Enhanced DGA features", X_enhanced),
    ]:
        scores = cross_validate(
            model,
            features,
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
            "accuracy": scores["test_accuracy"].mean(),
            "macro_f1": scores["test_macro_f1"].mean(),
            "macro_f1_std": scores["test_macro_f1"].std(),
            "balanced_accuracy":
                scores["test_balanced_accuracy"].mean(),
        })

    results_df = pd.DataFrame(results)

    print("\nRESULTS")
    print(results_df.to_string(index=False))

    improvement = (
        results_df.iloc[1]["macro_f1"]
        - results_df.iloc[0]["macro_f1"]
    )

    print(f"\nMacro-F1 difference: {improvement:+.4f}")


if __name__ == "__main__":
    main()
