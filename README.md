# Machine Learning Classification – Breast Cancer Wisconsin Diagnostic Dataset

## a. Problem Statement

The objective of this project is to develop and compare multiple Machine Learning classification models to predict whether a breast tumour is **Benign or Malignant** using the Breast Cancer Wisconsin Diagnostic Dataset.

The project implements five classification algorithms and evaluates their performance using different evaluation metrics. A Streamlit web application is also developed to allow users to upload test datasets, select a machine learning model and view prediction results.

## b. Dataset Description

**Dataset Name:** Breast Cancer Wisconsin Diagnostic Dataset

**Dataset Details:**
- Total Records: 569
- Total Features: 30
- Target Variable: Diagnosis
- Classification Type: Binary Classification
- Output Classes: Benign and Malignant

The dataset contains numerical features extracted from digitized images of fine needle aspirates (FNA) of breast masses.

The features describe characteristics of cell nuclei, including:
- Mean radius
- Mean texture
- Mean perimeter
- Mean area
- Mean smoothness
- Mean compactness
- Mean concavity
- Mean concave points
- Mean symmetry
- Mean fractal dimension
- Corresponding worst and standard error measurements.

The dataset is divided into training and testing datasets to train and evaluate the classification models.

The test dataset used in the experiments contains 114 records.

## c. GitHub Repository Link

**GitHub Repository:**

https://github.com/Pavithra-2203/ML-Assignment-2

The repository contains the complete source code, Jupyter Notebook, trained machine learning models, requirements file, test dataset and model comparison results.

## d. Models Used

Five machine learning classification algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Support Vector Machine (SVM)

### Model Comparison Table

The following table presents the evaluation metrics obtained from all five classification models.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on Model Performance

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 96.49% and an excellent AUC score of 0.9960. It demonstrated strong classification performance with a precision of 0.9750 and recall of 0.9286. It provides reliable and consistent predictions for this dataset.

**2. Decision Tree**

Decision Tree achieved an accuracy of 92.98%. It performed comparatively lower than the other models across most evaluation metrics. Its F1 score was 0.9048, indicating moderate classification performance.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 95.61% and an AUC score of 0.9823. It showed good precision of 0.9744 and an F1 score of 0.9383. The model performed well but had a slightly lower recall compared to Logistic Regression and SVM.

**4. Naive Bayes**

Naive Bayes achieved an accuracy of 93.86% and an AUC score of 0.9934. It obtained perfect precision of 1.0000, meaning that its positive predictions were correct in the test results. However, its recall of 0.8333 indicates that it missed some positive cases.

**5. Support Vector Machine (SVM)**

Support Vector Machine achieved the highest accuracy of 97.37% among all five models. It also obtained the highest F1 score of 0.9630 and MCC of 0.9442. Its precision was 1.0000 and recall was 0.9286. Overall, SVM demonstrated the strongest classification performance on this dataset.

### Overall Winner for the Dataset

**Support Vector Machine (SVM)** is the overall best-performing model for this dataset.

It achieved:
- Accuracy: 97.37%
- AUC: 0.9947
- Precision: 1.0000
- Recall: 0.9286
- F1 Score: 0.9630
- MCC: 0.9442

Based on the comparison of all five classification algorithms, SVM provides the best overall balance of classification performance and reliability for this experiment.

## Streamlit Web Application

The project includes an interactive Streamlit dashboard that allows users to:

- Select one of the five trained machine learning models.
- Upload a test dataset in CSV format.
- View the uploaded dataset.
- Generate classification predictions.
- View evaluation metrics.
- Display the classification report.

**Live Streamlit Application:** [Add your deployed Streamlit URL here]

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib
- Jupyter Notebook

## Project Files

- `ML_Assignment_2.ipynb` – Model development and experimentation.
- `app.py` – Streamlit application source code.
- `model/` – Trained machine learning model files.
- `model_comparison.csv` – Comparison of evaluation metrics.
- `test_data.csv` – Test dataset used in the experiments.
- `requirements.txt` – Required Python libraries.
- `README.md` – Project documentation.

## Conclusion

This project successfully implements and compares five machine learning classification algorithms using the Breast Cancer Wisconsin Diagnostic Dataset.

Among the five models, Support Vector Machine achieved the best overall results with 97.37% accuracy, a 0.9630 F1 score and a 0.9442 MCC.

The Streamlit dashboard provides an interactive interface for uploading datasets, selecting models and viewing classification results.

