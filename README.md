# Breast-Cancer-Gene-Expression-ML
Breast Cancer Gene Expression Analysis using Machine Learning (Logistic Regression, SVM, and Random Forest) with Python
# Breast Cancer Gene Expression Analysis Using Machine Learning

## Project Overview

This project applies machine learning techniques to classify breast cancer and normal tissue samples using gene expression data obtained from the NCBI Gene Expression Omnibus (GEO) database.

The project involves preprocessing gene expression data, selecting the most informative genes, training multiple machine learning models, and comparing their performance.

---

## Dataset

- Dataset: GSE15852
- Source: NCBI Gene Expression Omnibus (GEO)
- Organism: Homo sapiens
- Samples: 86 (43 Normal, 43 Tumor)
- Genes: 22,283
- Platform: Affymetrix Human Genome U133A Array

---

## Objectives

- Preprocess breast cancer gene expression data.
- Perform feature selection using SelectKBest.
- Train multiple machine learning models.
- Compare model performance.
- Identify important genes contributing to breast cancer classification.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- VS Code
- Anaconda

---

## Machine Learning Models

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

---

## Workflow

1. Download dataset from GEO
2. Data preprocessing
3. Feature matrix preparation
4. Train-test split
5. Feature selection (Top 100 genes)
6. Model training
7. Model evaluation
8. Performance comparison

---

## Results

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 94.44% |
| Support Vector Machine | 88.89% |
| Random Forest | 100.00% |

Random Forest achieved the highest classification accuracy for this dataset.

---

## Output

The project generates:

- Confusion Matrices
- ROC Curve
- Feature Importance Plot
- Model Comparison Graph
- CSV file containing model comparison results

---

## Future Improvements

- Deep Learning models
- Cross-validation
- Hyperparameter tuning
- External dataset validation

---

## Author

Mukesh Varathan A A

M.Sc. Microbiology

Bioinformatics | Machine Learning | Computational Biology
