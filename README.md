# Machine Learning Pipeline for Diabetic Nephropathy Prediction

## Overview
Diabetic Nephropathy (DN) is a severe complication of diabetes. This repository contains an end-to-end machine learning pipeline designed to predict the risk of DN using standard demographic and clinical laboratory data. 

The workflow is structured into five distinct phases, moving from rigorous data cleaning and imputation to clinically-driven feature engineering, advanced predictive modeling, and Explainable AI (XAI).

## Dataset
The project utilizes the `Diabetic_Nephropathy_v1.xlsx` dataset, which contains 767 patient records. The data encompasses 22 variables, including:
* **Demographics & Anthropometrics**: Age, Sex, Height, Weight, and BMI.
* **Vital Signs**: Systolic (SBP) and Diastolic Blood Pressure (DBP).
* **Laboratory Results**: HbA1c, Fasting Blood Glucose (FBG), Triglycerides (TG), C-peptide, Total Cholesterol (TC), HDL, and LDL.
* **Medical History & Treatments**: Diabetes duration, Diabetic Retinopathy, Smoking, Drinking, Insulin, Metformin, and Lipid-lowering drugs.
* **Target Variable**: Diabetic Nephropathy (DN).

## Project Workflow

### Phase 1: Exploratory Data Analysis (EDA) & Data Collection
* **Data Cleaning:** Standardization of column names and conversion of categorical variables (Sex) into numeric formats.
* **Outlier Handling:** Detection of outliers using the Interquartile Range (IQR). Clinically impossible values (e.g., DBP > 200 mmHg, LDLC > 20 mmol/L) were identified and safely converted to `NaN` for proper imputation downstream.
* **Target Analysis:** Assessment of class distribution, identifying a mild imbalance (approx. 74% negative vs. 26% positive for DN).
* **Correlation:** Pearson correlation matrices and horizontal bar charts isolated to show the specific predictive power of each feature against the DN target.
* **Distributions:** Visualized using individual boxplots for all clinical continuous variables.

### Phase 2: Preprocessing & Feature Engineering
* **Imputation:** Missing data in continuous clinical variables were imputed using **K-Nearest Neighbors (KNNImputer, k=5)** to maintain realistic multivariate patient profiles.
* **Feature Engineering:** Creation of highly relevant clinical indices derived from the base labs:
  * Pulse Pressure (PP)
  * Mean Arterial Pressure (MAP)
  * Non-HDL Cholesterol
  * LDL-to-HDL ratio
  * TyG index (Triglyceride-Glucose index)
* **Scaling:** Application of `StandardScaler` to ensure all continuous variables have zero mean and unit variance.
* **Multicollinearity Diagnostics:** Assessed using Pearson correlation and Variance Inflation Factor (VIF) to document highly correlated predictors before modeling.

### Phase 3: Generation of Models
The dataset was split into training (70%) and testing (30%) sets using stratified sampling to preserve the class imbalance ratio. 
Hyperparameter tuning was performed using `RandomizedSearchCV` (5-fold CV, optimizing for ROC-AUC) on the following models:
* Random Forest
* Support Vector Machine (SVM with RBF kernel)
* XGBoost
* Multi-Layer Perceptron (MLP) Neural Network

A **Stacking Ensemble Classifier** was also built, using the four tuned models as base estimators and a Logistic Regression model as the final meta-learner. For every model, an optimal decision threshold was mathematically determined to maximize the F1-score for the positive class.

### Phase 4: Model Comparison
Models were evaluated on the 30% hold-out test set. A comprehensive results matrix was generated to compare:
* ROC-AUC
* Sensitivity (Recall)
* Specificity
* Precision
* F1-score

A global ranking system was implemented to easily identify the best-performing models based on combined AUC and F1-score metrics.

### Phase 5: Interpretability & Results
* **Performance Visualizations:** ROC Curves and normalized Confusion Matrices were plotted for all models.
* **Explainable AI (SHAP):** The best-performing tree-based model (XGBoost) was evaluated using SHAP (SHapley Additive exPlanations) to ensure clinical transparency:
  * **Global Interpretability:** SHAP Summary plots and feature importance bar charts demonstrate which clinical variables drive the cohort's overall predictions.
  * **Local Interpretability:** SHAP Waterfall plots break down the exact risk factors and protective factors for individual patient predictions.

## Repository Files
* `Diabetic_Nephropathy_v1.xlsx`: The source dataset containing the clinical records.
* `Diabetic_Nephropathy_code.ipynb`: The Jupyter Notebook containing the interactive execution of the 5-phase pipeline, including plots and statistical outputs.
* `diabetic_nephropathy_code.py`: The Python script version of the complete machine learning workflow.
