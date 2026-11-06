
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ML Classification Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.title("Machine Learning Classification Dashboard")
st.write("Breast Cancer Wisconsin Diagnostic Dataset")

st.info("Upload your test CSV file and select an ML model.")

model_files = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

selected_model = st.selectbox(
    "Select ML Model",
    list(model_files.keys())
)

uploaded_file = st.file_uploader(
    "Upload Test Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")
    st.write("Total records:", len(df))
    st.dataframe(df.head())

    if "diagnosis" not in df.columns:
        st.error("Diagnosis column is missing.")

    else:

        X = df.drop(columns=["diagnosis"])

        mapping = {
            "benign": 0,
            "malignant": 1
        }

        if df["diagnosis"].dtype == "object":
            y_actual = (
                df["diagnosis"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(mapping)
            )
        else:
            y_actual = pd.to_numeric(
                df["diagnosis"],
                errors="coerce"
            )

        if y_actual.isna().any() or not set(y_actual.unique()).issubset({0, 1}):

            st.error("Invalid diagnosis values. Use benign/malignant or 0/1.")

        else:

            model_path = model_files[selected_model]

            if not os.path.exists(model_path):

                st.error("Model file not found: " + model_path)

            else:

                try:

                    model = joblib.load(model_path)

                    # Predict without retraining
                    y_pred = model.predict(X)
                    y_prob = model.predict_proba(X)[:, 1]

                    # Calculate metrics
                    accuracy = accuracy_score(y_actual, y_pred)
                    auc = roc_auc_score(y_actual, y_prob)
                    precision = precision_score(
                        y_actual, y_pred, zero_division=0
                    )
                    recall = recall_score(
                        y_actual, y_pred, zero_division=0
                    )
                    f1 = f1_score(
                        y_actual, y_pred, zero_division=0
                    )
                    mcc = matthews_corrcoef(y_actual, y_pred)

                    st.subheader("Evaluation Metrics")

                    c1, c2, c3 = st.columns(3)

                    c1.metric("Accuracy", f"{accuracy:.4f}")
                    c2.metric("AUC Score", f"{auc:.4f}")
                    c3.metric("Precision", f"{precision:.4f}")

                    c4, c5, c6 = st.columns(3)

                    c4.metric("Recall", f"{recall:.4f}")
                    c5.metric("F1 Score", f"{f1:.4f}")
                    c6.metric("MCC", f"{mcc:.4f}")

                    # Confusion matrix
                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(
                        y_actual,
                        y_pred,
                        labels=[0, 1]
                    )

                    fig, ax = plt.subplots(figsize=(6, 4))

                    sns.heatmap(
                        cm,
                        annot=True,
                        fmt="d",
                        cmap="Blues",
                        xticklabels=["Benign", "Malignant"],
                        yticklabels=["Benign", "Malignant"],
                        ax=ax
                    )

                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")

                    st.pyplot(fig)
                    plt.close(fig)

                    # Classification report
                    st.subheader("Classification Report")

                    report = classification_report(
                        y_actual,
                        y_pred,
                        labels=[0, 1],
                        target_names=["Benign", "Malignant"],
                        output_dict=True,
                        zero_division=0
                    )

                    st.dataframe(
                        pd.DataFrame(report).transpose().round(4)
                    )

                    # Prediction results
                    st.subheader("Prediction Results")

                    result_df = df.copy()

                    result_df["Predicted Diagnosis"] = [
                        "Malignant" if p == 1 else "Benign"
                        for p in y_pred
                    ]

                    st.dataframe(result_df.head(20))

                    st.download_button(
                        "Download Predictions CSV",
                        result_df.to_csv(index=False),
                        "prediction_results.csv",
                        "text/csv"
                    )

                    st.success("Evaluation completed successfully!")

                except Exception as error:

                    st.error("Error during model evaluation:")
                    st.exception(error)
