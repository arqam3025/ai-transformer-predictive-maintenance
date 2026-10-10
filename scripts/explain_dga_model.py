
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from transformer_pm.dga import DGASample, engineer_features


DATASET = Path("data/processed/dga_589_ml.csv")
GASES = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]

FEATURES = GASES + [
    "RATIO_CH4_H2",
    "RATIO_C2H2_C2H4",
    "RATIO_C2H4_C2H6",
]

RANDOM_STATE = 42


def main():
    df = pd.read_csv(DATASET)

    X = df[GASES]
    y = df["fault_class"]

    # Keep the original holdout excluded.
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Separate training data into model-fit and explanation sets.
    X_fit, X_explain, y_fit, y_explain = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    def transform_features(data):
        records = []
        for row in data.itertuples(index=False, name=None):
            sample = DGASample(*[float(value) for value in row])
            records.append(engineer_features(sample))
        return pd.DataFrame(records, index=data.index)[FEATURES]

    X_fit = transform_features(X_fit)
    X_explain = transform_features(X_explain)

    model = RandomForestClassifier(
        n_estimators=500,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_fit, y_fit)

    predictions = model.predict(X_explain)

    print("=" * 65)
    print("EXPLAINABLE AI — DGA RANDOM FOREST")
    print("=" * 65)
    print(f"Model-fit samples: {len(X_fit)}")
    print(f"Explanation samples: {len(X_explain)}")
    print(
        "Explanation macro-F1:",
        round(f1_score(y_explain, predictions, average="macro"), 4),
    )

    importance = permutation_importance(
        model,
        X_explain,
        y_explain,
        scoring="f1_macro",
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

    results = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": importance.importances_mean,
        "importance_std": importance.importances_std,
    }).sort_values("importance_mean", ascending=False)

    print("\nPERMUTATION FEATURE IMPORTANCE")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
