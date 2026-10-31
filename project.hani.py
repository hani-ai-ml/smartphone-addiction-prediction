"""SUCHAK HANI P  DIV:5  ROLL NO : 554"""
#Smartphone Addiction Prediction - complete supervised ML classification project.
from __future__ import annotations
import json
import os
import webbrowser
import warnings
from pathlib import Path

# Keep Matplotlib's cache inside the project so the script works in restricted VS Code setups too.
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_ROOT / ".matplotlib"))

import joblib
import matplotlib
#matplotlib.use("Agg")  # Non-interactive backend: reliably saves charts from the VS Code terminal.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------- 1. Project paths and settings --------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
ROOT = SCRIPT_ROOT
DOCUMENTS_PATH = Path(os.path.expanduser("~")) / "Documents" / "Phone_addication.csv"
DATA_PATH = DOCUMENTS_PATH if DOCUMENTS_PATH.exists() else ROOT / "Dataset" / "Phone_addication.csv"
REPORTS_DIR = ROOT / "Reports"
MODELS_DIR = ROOT / "models"
REPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TARGET = "addicted_label"
ID_COLUMN = "user_id"
# addiction_level represents the severity label and can directly reveal addicted_label.
# It is therefore intentionally excluded from model input to prevent target leakage.
LEAKAGE_COLUMN = "addiction_level"


def section(title: str) -> None:
    """Print a clear heading in the VS Code terminal."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    
# ==============================================================================
# FINAL MODEL OUTPUT: ADDICTION STATUS FUNCTION
# ==============================================================================
def print_addiction_summary(metrics: dict, test_predictions_path: Path) -> None:
    print("\n" + "=" * 78)
    print("         FINAL SMARTPHONE ADDICTION PREDICTION SUMMARY REPORT           ")
    print("=" * 78)
    
    # Accuracy and metrics print
    acc = metrics.get("test_accuracy", 0.0) * 100
    f1 = metrics.get("test_f1_score", 0.0)
    print(f"► Model Name                : {metrics.get('final_model', 'Tuned Model')}")
    print(f"► Model Test Accuracy       : {acc:.2f}%")
    print(f"► Model F1-Score            : {f1:.4f}")
    
    # Read predictions to calculate addiction count
    if test_predictions_path.exists():
        df_pred = pd.read_csv(test_predictions_path)
        total_users = len(df_pred)
        addicted_count = (df_pred["predicted_addicted_label"] == 1).sum()
        not_addicted_count = total_users - addicted_count
        addiction_pct = (addicted_count / total_users) * 100 if total_users > 0 else 0
        
        print("-" * 78)
        print(f"► Total Tested Users        : {total_users}")
        print(f"► Predicted ADDICTED (1) ⚠️  : {addicted_count} users ({addiction_pct:.1f}%)")
        print(f"► Predicted NOT ADDICTED (0) ✅: {not_addicted_count} users ({100 - addiction_pct:.1f}%)")
    
    print("=" * 78 + "\n")


def save_figure(filename: str) -> None:
    """Save the current Matplotlib figure in the Reports folder."""
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / filename, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close()


def make_encoder(dense: bool = False) -> OneHotEncoder:
    """Create an encoder compatible with both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=not dense)


# -------------------------- 2. Import and understand data --------------------------
def load_and_understand_data() -> pd.DataFrame:
    section("STEP 1: DATA IMPORT AND UNDERSTANDING")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn data types:\n", df.dtypes)
    print("\nFirst five records:\n", df.head())

    summary = pd.DataFrame(
        {
            "data_type": df.dtypes.astype(str),
            "missing_values": df.isna().sum(),
            "unique_values": df.nunique(),
        }
    )
    summary.to_csv(REPORTS_DIR / "dataset_summary.csv")
    print("\nMissing values by column:\n", df.isna().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\nTarget distribution:\n", df[TARGET].value_counts().sort_index())
    return df


# -------------------------- 3. Missing values and outlier treatment --------------------------
def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    section("STEP 2: DATA PREPROCESSING")
    clean = df.copy()
    duplicates_before = int(clean.duplicated().sum())
    clean = clean.drop_duplicates().reset_index(drop=True)

    numeric_columns = clean.select_dtypes(include=np.number).columns.drop(TARGET).tolist()
    categorical_columns = clean.select_dtypes(include="object").columns.tolist()

    # Missing values: numeric columns use median, categorical columns use mode.
    for col in numeric_columns:
        clean[col] = clean[col].fillna(clean[col].median())
    for col in categorical_columns:
        mode = clean[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        clean[col] = clean[col].fillna(fill_value)

    # IQR capping retains all records while limiting extreme numeric outliers.
    outlier_rows = []
    for col in numeric_columns:
        q1, q3 = clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        before = int(((clean[col] < lower) | (clean[col] > upper)).sum())
        clean[col] = clean[col].clip(lower=lower, upper=upper)
        after = int(((clean[col] < lower) | (clean[col] > upper)).sum())
        outlier_rows.append(
            {"column": col, "lower_limit": lower, "upper_limit": upper,
             "outliers_before": before, "outliers_after": after}
        )

    outlier_report = pd.DataFrame(outlier_rows)
    outlier_report.to_csv(REPORTS_DIR / "outlier_before_after.csv", index=False)
    clean.to_csv(REPORTS_DIR / "cleaned_dataset.csv", index=False)
    print(f"Duplicates removed: {duplicates_before}")
    print("Missing values after treatment:", int(clean.isna().sum().sum()))
    print("Outlier comparison saved as Reports/outlier_before_after.csv")
    return clean, outlier_report


# -------------------------- 4. Exploratory Data Analysis --------------------------
def create_eda(df: pd.DataFrame) -> None:
    section("STEP 3: EXPLORATORY DATA ANALYSIS AND VISUALIZATION")
    sns.set_theme(style="whitegrid", palette="deep")
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    numeric_without_target = [c for c in numeric if c != TARGET]

    # Histogram - distribution of all numeric features.
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, col in zip(axes.flat, numeric_without_target):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, element="step", ax=ax)
        ax.set_title(f"Distribution of {col}")
    for ax in axes.flat[len(numeric_without_target):]:
        ax.set_visible(False)
    save_figure("01_histograms.png")

    # Box plot - shows outliers and class-wise feature distributions.
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, col in zip(axes.flat, numeric_without_target):
        sns.boxplot(data=df, x=TARGET, y=col, ax=ax)
        ax.set_title(f"{col} by addiction class")
    for ax in axes.flat[len(numeric_without_target):]:
        ax.set_visible(False)
    save_figure("02_boxplots.png")

    # Pie chart - required class balance visualization.
    counts = df[TARGET].value_counts().sort_index()
    plt.figure(figsize=(7, 6))
    plt.pie(counts, labels=["Not Addicted (0)", "Addicted (1)"], autopct="%1.1f%%", startangle=90)
    plt.title("Addiction Class Distribution")
    save_figure("03_target_pie_chart.png")

    # Bar charts - categorical relationship with the target.
    categorical = ["gender", "stress_level", "academic_work_impact"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, col in zip(axes, categorical):
        sns.countplot(data=df, x=col, hue=TARGET, ax=ax)
        ax.set_title(f"{col} versus addiction label")
        ax.tick_params(axis="x", rotation=25)
    save_figure("04_categorical_bar_charts.png")

    # Correlation matrix - numeric columns plus binary target.
    plt.figure(figsize=(12, 9))
    corr = df[numeric].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Matrix")
    save_figure("05_correlation_matrix.png")

    # Severity analysis is informative but not used as a feature because it leaks target information.
    severity = pd.crosstab(df[LEAKAGE_COLUMN], df[TARGET], normalize="index") * 100
    severity.to_csv(REPORTS_DIR / "addiction_severity_analysis.csv")
    print("Saved histograms, box plots, pie chart, bar charts, and correlation matrix.")


def create_automated_eda_report(df: pd.DataFrame) -> None:
    """Generate the required automated EDA HTML report using ydata-profiling when available."""
    section("STEP 4: AUTOMATED EDA REPORT")
    output = REPORTS_DIR / "EDA_Report.html"
    try:
        from ydata_profiling import ProfileReport

        profile = ProfileReport(df, title="Smartphone Addiction - Automated EDA", explorative=True, minimal=True)
        profile.to_file(output)
        print("ydata-profiling report created successfully.")
    except Exception as error:
        # The fallback keeps the project runnable even if optional profiling is unavailable.
        tables = (
            "<h1>Smartphone Addiction - Automated Data Summary</h1>"
            "<h2>Dataset preview</h2>" + df.head(20).to_html(index=False) +
            "<h2>Descriptive statistics</h2>" + df.describe(include="all").T.to_html() +
            "<h2>Missing values</h2>" + df.isna().sum().to_frame("missing_values").to_html() +
            f"<p>ydata-profiling was unavailable: {error}</p>"
        )
        output.write_text(tables, encoding="utf-8")
        print("Fallback HTML data summary created. Install requirements to generate full ydata report.")
# HTML રિપોર્ટને ડિફોલ્ટ બ્રાઉઝરમાં ઓપન કરશે
    if output.exists():
        webbrowser.open(output.as_uri())


# -------------------------- 5. Feature selection and supervised model --------------------------
def prepare_features(df: pd.DataFrame):
    section("STEP 5: FEATURE SELECTION AND TRAIN-TEST SPLIT")
    # user_id is an identifier; addiction_level is excluded to avoid target leakage.
    feature_columns = [c for c in df.columns if c not in [TARGET, ID_COLUMN, LEAKAGE_COLUMN]]
    X = df[feature_columns].copy()
    y = df[TARGET].astype(int).copy()
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    feature_report = pd.DataFrame(
        {
            "feature": feature_columns,
            "type": ["Numerical" if c in numeric_features else "Categorical" for c in feature_columns],
            "used_in_model": True,
            "reason": "Predictive input variable",
        }
    )
    excluded = pd.DataFrame(
        [
            {"feature": ID_COLUMN, "type": "Identifier", "used_in_model": False,
             "reason": "Unique user ID has no behavioral meaning"},
            {"feature": LEAKAGE_COLUMN, "type": "Leakage risk", "used_in_model": False,
             "reason": "Severity category directly reveals the binary addiction outcome"},
            {"feature": TARGET, "type": "Target", "used_in_model": False,
             "reason": "Dependent variable to predict"},
        ]
    )
    pd.concat([feature_report, excluded], ignore_index=True).to_csv(
        REPORTS_DIR / "feature_selection_report.csv", index=False
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Independent variables: {feature_columns}")
    print(f"Dependent variable: {TARGET}")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


def build_preprocessor(numeric_features, categorical_features, dense: bool = False) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", make_encoder(dense))]), categorical_features),
        ],
        remainder="drop",
    )


def choose_and_train_model(X_train, y_train, numeric_features, categorical_features):
    section("STEP 6: MODEL BUILDING, CROSS-VALIDATION AND IMPROVEMENT")
    # Required baseline: Logistic Regression. Additional models are compared fairly using identical folds.
    sparse_preprocessor = build_preprocessor(numeric_features, categorical_features, dense=False)
    dense_preprocessor = build_preprocessor(numeric_features, categorical_features, dense=True)
    candidates = {
        "Logistic Regression (baseline)": Pipeline([
            ("preprocessor", sparse_preprocessor),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", sparse_preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocessor", dense_preprocessor),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipeline in candidates.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "f1": "f1", "precision": "precision", "recall": "recall"},
            n_jobs=-1,
        )
        rows.append({
            "model": name,
            "cv_accuracy_mean": scores["test_accuracy"].mean(),
            "cv_accuracy_std": scores["test_accuracy"].std(),
            "cv_f1_mean": scores["test_f1"].mean(),
            "cv_precision_mean": scores["test_precision"].mean(),
            "cv_recall_mean": scores["test_recall"].mean(),
        })
    comparison = pd.DataFrame(rows).sort_values("cv_accuracy_mean", ascending=False).reset_index(drop=True)
    comparison.to_csv(REPORTS_DIR / "model_comparison_cross_validation.csv", index=False)
    print("\n5-fold cross-validation comparison:\n", comparison.round(4))

    # Regularization: C controls Logistic Regression regularization. Hyperparameter tuning finds the best setting.
    logistic_tuning = GridSearchCV(
        candidates["Logistic Regression (baseline)"],
        param_grid={"model__C": [0.01, 0.1, 1, 10, 100]},
        scoring="accuracy", cv=cv, n_jobs=-1, refit=True,
    )
    logistic_tuning.fit(X_train, y_train)

    forest_tuning = GridSearchCV(
        candidates["Random Forest"],
        param_grid={
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        },
        scoring="accuracy", cv=cv, n_jobs=-1, refit=True,
    )
    forest_tuning.fit(X_train, y_train)

    tuning_results = pd.DataFrame([
        {"model": "Regularized Logistic Regression", "best_cv_accuracy": logistic_tuning.best_score_, "best_parameters": str(logistic_tuning.best_params_)},
        {"model": "Tuned Random Forest", "best_cv_accuracy": forest_tuning.best_score_, "best_parameters": str(forest_tuning.best_params_)},
    ]).sort_values("best_cv_accuracy", ascending=False)
    tuning_results.to_csv(REPORTS_DIR / "hyperparameter_tuning_results.csv", index=False)

    best_name = tuning_results.iloc[0]["model"]
    best_model = logistic_tuning.best_estimator_ if best_name.startswith("Regularized") else forest_tuning.best_estimator_
    print(f"\nSelected final model by highest cross-validation accuracy: {best_name}")
    print("Best parameters:", best_model.named_steps["model"].get_params())
    return best_model, best_name, comparison, tuning_results


# -------------------------- 6. Model evaluation, predictions and outputs --------------------------
def evaluate_model(model, model_name: str, X_test, y_test) -> dict:
    section("STEP 7: FINAL MODEL EVALUATION")
    predictions = model.predict(X_test)
    metrics = {
        "final_model": model_name,
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1_score": float(f1_score(y_test, predictions, zero_division=0)),
        "test_mse": float(mean_squared_error(y_test, predictions)),
        "test_rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
    }
    # MSE/RMSE are supplementary binary-error measures. R2 is not used because this is classification.
    metrics["note"] = "Classification task: Accuracy, Precision, Recall and F1 are primary metrics. MSE/RMSE are supplementary binary prediction errors."
    (REPORTS_DIR / "final_metrics.json").write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print("\nFinal test metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

    report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True, zero_division=0)).transpose()
    report.to_csv(REPORTS_DIR / "classification_report.csv")
    print("\nClassification report:\n", classification_report(y_test, predictions, zero_division=0))

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Addicted", "Addicted"], yticklabels=["Not Addicted", "Addicted"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    save_figure("06_confusion_matrix.png")

    prediction_output = X_test.copy()
    prediction_output["actual_addicted_label"] = y_test.values
    prediction_output["predicted_addicted_label"] = predictions
    if hasattr(model, "predict_proba"):
        prediction_output["addiction_probability"] = model.predict_proba(X_test)[:, 1]
    prediction_output.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    # Feature importance makes the selected Random Forest explainable.
    if hasattr(model.named_steps["model"], "feature_importances_"):
        transformed_names = model.named_steps["preprocessor"].get_feature_names_out()
        importance = pd.DataFrame({"feature": transformed_names, "importance": model.named_steps["model"].feature_importances_})
        importance = importance.sort_values("importance", ascending=False).head(15)
        importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
        plt.figure(figsize=(10, 7))
        sns.barplot(data=importance, x="importance", y="feature", color="#377eb8")
        plt.title("Top 15 Feature Importances")
        save_figure("07_feature_importance.png")

    joblib.dump(model, MODELS_DIR / "smartphone_addiction_model.joblib")
    return metrics


def main() -> None:
    original_df = load_and_understand_data()
    cleaned_df, _ = clean_data(original_df)
    create_eda(cleaned_df)
    create_automated_eda_report(cleaned_df)
    X_train, X_test, y_train, y_test, numerical, categorical = prepare_features(cleaned_df)
    model, model_name, _, _ = choose_and_train_model(X_train, y_train, numerical, categorical)
    metrics = evaluate_model(model, model_name, X_test, y_test)

    # Secondary supervised regression task: predict a continuous usage outcome.
    section("STEP 8: SUPERVISED REGRESSION - DAILY SCREEN TIME PREDICTION")
    from regression_model import run_regression_model
    regression_metrics = run_regression_model(cleaned_df, ROOT)
    print(f"Regression target: {regression_metrics['regression_target']}")
    print(f"Regression R2: {regression_metrics['test_r2']:.4f}")
    print(f"Regression RMSE: {regression_metrics['test_rmse']:.4f}")
    print(f"Regression target: {regression_metrics['regression_target']}")
    print(f"Regression R2: {regression_metrics['test_r2']:.4f}")
    print(f"Regression RMSE: {regression_metrics['test_rmse']:.4f}")
    
    print_addiction_summary(metrics, REPORTS_DIR / "test_predictions.csv")

    # Create the report only after charts, metrics and model outputs exist.
    from build_report import create_project_report
    create_project_report(ROOT, metrics, regression_metrics)

    section("PROJECT COMPLETED SUCCESSFULLY")
    print(f"All charts, EDA HTML report, result files and PDF report are in: {REPORTS_DIR}")
    print(f"Saved model is in: {MODELS_DIR / 'smartphone_addiction_model.joblib'}")

    # PDF રિપોર્ટ અને Reports ફોલ્ડર આપમેળે સિસ્ટમમાં ઓપન કરવા માટે
    pdf_path = REPORTS_DIR / "Smartphone_Addiction_Report.pdf" # તમારા PDF નું સાચું નામ ચકાસી લેવું
    if pdf_path.exists():
        os.startfile(pdf_path)  # Windows માં PDF સફેરી/એજ/એક્રોબેટમાં ઓપન કરશે

    # આખું Reports ફોલ્ડર ફાઈલ એક્સપ્લોરરમાં ઓપન કરવા માટે:
    os.startfile(REPORTS_DIR)
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
