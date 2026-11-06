
# Customer Churn Prediction Using Artificial Neural Networks (ANN)

An end-to-end deep learning project using **TensorFlow/Keras** to predict customer churn from demographic, account, service usage, and satisfaction data.

The project covers exploratory data analysis (EDA), preprocessing, ANN development, model optimization, and evaluation on unseen customer records.

## Problem Statement

Customer churn occurs when customers stop using a company's products or services. Predicting churn can help businesses identify potentially at-risk customers and explore retention strategies.

**Objective:** Develop and evaluate an ANN-based binary classification model that estimates customer churn probability.

## Dataset

- **Records:** 180 customers
- **Input Features:** 11
- **Target:** Churn (Yes = 1, No = 0)
- **Class Distribution:** 94 churned, 86 retained
- **Data Split:** 70% training, 15% validation, 15% testing

The dataset includes demographic attributes, customer tenure, monthly charges, support interactions, data usage, satisfaction scores, and categorical account details.

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib
- Jupyter Notebook

## Project Workflow

1. Dataset inspection and data quality assessment
2. Exploratory Data Analysis (EDA)
3. Numerical and categorical feature analysis
4. Outlier detection using IQR
5. Stratified train-validation-test splitting
6. StandardScaler and OneHotEncoder preprocessing
7. ANN architecture design and training
8. Overfitting analysis and L2 regularization experiment
9. Validation-based threshold optimization
10. Final test evaluation and baseline comparison
11. Saving the selected model and preprocessing artifacts

## ANN Architecture

**Selected model:**

Input (22 features) → Dense (16, ReLU) → Dense (8, ReLU) → Dense (1, Sigmoid)

- **Trainable Parameters:** 513
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **EarlyStopping:** Enabled
- **Selected Classification Threshold:** 0.30

A smaller L2-regularized ANN was also evaluated, but the original architecture achieved a higher validation F1-score.

## Final Test Performance

| Metric | Score |
|---|---:|
| Accuracy | 51.85% |
| Precision | 52.38% |
| Recall | 78.57% |
| F1-Score | 62.86% |
| ROC-AUC | 0.4780 |

**Confusion Matrix:**
- True Positives: 11
- True Negatives: 3
- False Positives: 10
- False Negatives: 3

## Model Assessment

The ANN identified 11 of 14 actual churners in the test dataset. However, its accuracy matched the majority-class baseline, while its F1-score and ROC-AUC were lower.

The dataset contains only 180 records, limiting model generalization and confidence in the evaluation results.

**This project is an educational deep learning experiment and is not recommended for production deployment.**

## Project Structure

```text
Deep_Learning_ANN/
├── ANN_Customer_Churn.ipynb
├── cleaned_customerchurn_dataset.csv
├── .gitignore
├── README.md
└── churn_ann_artifacts/
    ├── customer_churn_ann.keras
    ├── preprocessor.joblib
    └── model_settings.json
```

## Future Improvements

- Train and evaluate models using larger, representative datasets.
- Compare the ANN with Logistic Regression and other baseline classifiers.
- Evaluate model stability using repeated stratified cross-validation.
- Improve feature engineering using verified customer behavior information.
- Assess probability calibration and practical intervention costs.

## Conclusion

This project demonstrates an end-to-end ANN workflow, from data understanding through model evaluation and artifact saving.

The experiment highlights the importance of detecting overfitting, selecting thresholds using validation data, comparing against simple baselines, and reporting model limitations transparently.

