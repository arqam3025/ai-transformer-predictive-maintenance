"""Train baseline transformer DGA fault-classification models."""

from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)

DATASET = Path("data/processed/dga_589_ml.csv")

FEATURES = ["H2", "CH4", "C2H6", "C2H4", "C2H2"]
TARGET = "fault_class"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_ml_data():
    """Load features and target from the cleaned DGA dataset."""

    df = pd.read_csv(DATASET)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train a model and calculate evaluation metrics."""

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    results = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
        ),
    }

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    print(
        f"Accuracy:          "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Balanced accuracy: "
        f"{results['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{results['macro_f1']:.4f}"
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return results


def main():
    X, y = load_ml_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("=" * 65)
    print("TRANSFORMER DGA FAULT CLASSIFICATION")
    print("=" * 65)

    print(f"\nTotal samples: {len(X)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    print("\nTraining class distribution:")
    print(y_train.value_counts().sort_index().to_string())

    print("\nTest class distribution:")
    print(y_test.value_counts().sort_index().to_string())

    models = {
        "Dummy baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Random forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "Histogram gradient boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for name, model in models.items():
        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df
        .sort_values("macro_f1", ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()