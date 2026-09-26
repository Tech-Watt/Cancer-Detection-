"""Train and evaluate a cancer risk-level classifier, then save artifacts for the API."""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from eda import FIGURES_DIR, LEVEL_ORDER, ROOT, load_data, prepare_xy, save_fig

MODELS_DIR = ROOT / "models"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def plot_evaluation(y_true, y_pred, labels: list[str], importances: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title("Confusion matrix (test set)")
    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("True risk")
    save_fig("05_confusion_matrix.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 7))
    top = importances.sort_values(ascending=True).tail(15)
    ax.barh(top.index, top.values, color="#2C6EAB")
    ax.set_title("Top 15 features used by the model")
    ax.set_xlabel("Random Forest importance")
    save_fig("06_feature_importance.png")
    plt.close()


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y, feature_cols = prepare_xy(df)

    encoder = LabelEncoder()
    encoder.fit(LEVEL_ORDER)
    encoder.classes_ = np.array(LEVEL_ORDER)
    y_encoded = np.array([LEVEL_ORDER.index(label) for label in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    candidates = build_models()
    comparison = {}
    fitted = {}
    for name, pipeline in candidates.items():
        cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_macro")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        metrics = evaluate(y_test, pred, labels=list(range(len(LEVEL_ORDER))))
        comparison[name] = {
            "cv_f1_macro_mean": float(cv_f1.mean()),
            "cv_f1_macro_std": float(cv_f1.std()),
            "test_accuracy": metrics["accuracy"],
            "test_f1_macro": metrics["f1_macro"],
        }
        fitted[name] = (pipeline, pred, metrics)

    # Prefer the forest when scores tie so the API can return feature importances.
    best_name = max(
        comparison,
        key=lambda k: (comparison[k]["test_f1_macro"], 1 if k == "random_forest" else 0),
    )
    best_pipeline, best_pred, best_metrics = fitted[best_name]

    forest = fitted["random_forest"][0].named_steps["model"]
    importances = pd.Series(forest.feature_importances_, index=feature_cols)
    plot_evaluation(y_test, best_pred, labels=list(range(len(LEVEL_ORDER))), importances=importances)

    artifact = {
        "pipeline": best_pipeline,
        "label_encoder": encoder,
        "feature_names": feature_cols,
        "class_names": LEVEL_ORDER,
        "model_name": best_name,
        "metrics": {
            "best_model": best_name,
            "comparison": comparison,
            "test": {
                "accuracy": best_metrics["accuracy"],
                "precision_macro": best_metrics["precision_macro"],
                "recall_macro": best_metrics["recall_macro"],
                "f1_macro": best_metrics["f1_macro"],
                "classification_report": {
                    LEVEL_ORDER[int(k)] if str(k).isdigit() else k: v
                    for k, v in best_metrics["classification_report"].items()
                },
                "confusion_matrix": best_metrics["confusion_matrix"],
            },
        },
    }
    joblib.dump(artifact, MODELS_DIR / "cancer_risk_model.joblib")
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(artifact["metrics"], f, indent=2)

    print("Best model:", best_name)
    print(json.dumps(comparison, indent=2))
    print("Test accuracy:", round(best_metrics["accuracy"], 4))
    print("Test macro F1:", round(best_metrics["f1_macro"], 4))
    print("Saved model to", MODELS_DIR / "cancer_risk_model.joblib")


if __name__ == "__main__":
    main()
