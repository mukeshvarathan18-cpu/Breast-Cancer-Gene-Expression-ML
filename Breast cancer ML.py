import pandas as pd

# To extract file
file_path = "C:/Users/HP/Desktop/GSE15852_series_matrix.txt/GSE15852_series_matrix.txt"

# Read all lines
with open(file_path, "r") as f:
    lines = f.readlines()

# total length of lines in the data
print("Total lines:", len(lines))
print("\nFirst 20 lines:\n")

for line in lines[:20]:
    # removes unnecessary spaces and newline characters
    print(line.strip())

# enumerate() gives both line number and line content
for i, line in enumerate(lines):
    if "!series_matrix_table_begin" in line:
        print("Table Begin:", i)

    if '"ID_REF"' in line:
        print("ID_REF:", i)

# Load only the expression matrix
df = pd.read_csv(
    file_path,
    sep="\t",  # separated using tabs
    skiprows=78, # To ignore first 78 lines
    skipfooter=1, # removing those 78 lines
    engine="python"
)


print(df.head()) # first 5 lines
print(df.shape) # columns and rows

# Basic information about the dataset

# Display all column names in the dataset
print("\nColumn Names:\n")
print(df.columns)

# Display the data type of every column
print("\nData Types:\n")
print(df.dtypes)

# Display dataset structure and summary information
print("\nDataset Information:\n")
df.info()

# Display descriptive statistics for numerical columns
print("\nStatistical Summary:\n")
print(df.describe())

# Set gene IDs as the index
df = df.set_index("ID_REF")

# Transpose the dataset
df_transposed = df.T

print(df_transposed.head())

print("\nShape after transpose:")
print(df_transposed.shape)

# Extract sample metadata from the GEO file
# Display sample characteristics for understanding class labels
for line in lines:
    if "!Sample_source_name_ch1" in line or "!Sample_characteristics_ch1" in line:
        print(line)


# Create the feature matrix
X = df_transposed.copy()

print("Feature Matrix Shape:")
print(X.shape)

print("\nFirst 5 samples:")
print(X.head())

# Find sample metadata

for line in lines:
    if "!Sample_title" in line:
        print(line)

    if "!Sample_source_name_ch1" in line:
        print(line)

    if "!Sample_characteristics_ch1" in line:
        print(line)

# Create target labels
y = []

for i in range(len(X)):
    if i % 2 == 0:
        y.append(0)      # Normal
    else:
        y.append(1)      # Tumor

print("Total labels:", len(y))
print("First 20 labels:")
print(y[:20])

# Import the train-test splitting function
from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split the dataset into training and testing sets
print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)


print("Training labels:", len(y_train))
print("Testing labels:", len(y_test))

from sklearn.feature_selection import SelectKBest, f_classif

# Select the top 100 most informative genes
selector = SelectKBest(score_func=f_classif, k=100)

# Fit the selector on the training data and transform it
X_train_selected = selector.fit_transform(X_train, y_train)

# Apply the same selected features to the test data
X_test_selected = selector.transform(X_test)

# Display dataset dimensions before and after feature selection
print("Original Training Shape:", X_train.shape)
print("Selected Training Shape:", X_train_selected.shape)

# Import Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create the model
lr_model = LogisticRegression(max_iter=1000)

# Train the model
lr_model.fit(X_train_selected, y_train)

# Predict on test data
y_pred = lr_model.predict(X_test_selected)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.svm import SVC

# Create SVM model
svm_model = SVC(kernel="linear", random_state=42)

# Train
svm_model.fit(X_train_selected, y_train)

# Predict
svm_pred = svm_model.predict(X_test_selected)

# Evaluate
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))

print("\nClassification Report")
print(classification_report(y_test, svm_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, svm_pred))

from sklearn.ensemble import RandomForestClassifier

# Create the Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
rf_model.fit(X_train_selected, y_train)

# Predict on test data
rf_pred = rf_model.predict(X_test_selected)

# Evaluate the model
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Logistic Regression
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# SVM
ConfusionMatrixDisplay.from_predictions(y_test, svm_pred)
plt.title("Support Vector Machine - Confusion Matrix")
plt.show()

# Random Forest
ConfusionMatrixDisplay.from_predictions(y_test, rf_pred)
plt.title("Random Forest - Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

# Logistic Regression probabilities
lr_prob = lr_model.predict_proba(X_test_selected)[:, 1]

# Random Forest probabilities
rf_prob = rf_model.predict_proba(X_test_selected)[:, 1]

# SVM with probability enabled
svm_model_prob = SVC(kernel="linear", probability=True, random_state=42)

svm_model_prob.fit(X_train_selected, y_train)

svm_prob = svm_model_prob.predict_proba(X_test_selected)[:, 1]

print("Logistic Regression probabilities:")
print(lr_prob[:5])

print("\nSVM probabilities:")
print(svm_prob[:5])

print("\nRandom Forest probabilities:")
print(rf_prob[:5])

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Logistic Regression ROC
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_prob)
lr_auc = roc_auc_score(y_test, lr_prob)

# SVM ROC
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_prob)
svm_auc = roc_auc_score(y_test, svm_prob)

# Random Forest ROC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_prob)
rf_auc = roc_auc_score(y_test, rf_prob)

# Plot ROC Curves
plt.figure(figsize=(8,6))

plt.plot(lr_fpr, lr_tpr,
         label=f"Logistic Regression (AUC = {lr_auc:.2f})")

plt.plot(svm_fpr, svm_tpr,
         label=f"SVM (AUC = {svm_auc:.2f})")

plt.plot(rf_fpr, rf_tpr,
         label=f"Random Forest (AUC = {rf_auc:.2f})")

# Random guessing line
plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend(loc="lower right")

plt.grid(True)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Get selected gene names
selected_gene_names = X_train.columns[selector.get_support()]

# Get feature importance from Random Forest
importance = rf_model.feature_importances_

# Create a DataFrame
feature_importance = pd.DataFrame({
    "Gene": selected_gene_names,
    "Importance": importance
})

# Sort by importance
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# Display top 20 genes
print("\nTop 20 Important Genes:\n")
print(feature_importance.head(20))

# Plot Top 20 genes
top20 = feature_importance.head(20)

plt.figure(figsize=(10,8))

plt.barh(top20["Gene"], top20["Importance"])

plt.xlabel("Feature Importance")
plt.ylabel("Gene")

plt.title("Top 20 Important Genes - Random Forest")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

import matplotlib.pyplot as plt

# Model names
models = [
    "Logistic Regression",
    "SVM",
    "Random Forest"
]

svm_pred = svm_model.predict(X_test_selected)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", svm_accuracy)

print(classification_report(y_test, svm_pred))

print(confusion_matrix(y_test, svm_pred))

# Their accuracies
accuracies = [
    accuracy,
    svm_accuracy,
    rf_accuracy
]

# Create the plot
plt.figure(figsize=(8,5))

bars = plt.bar(models, accuracies)

# Add accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f"{height:.2f}",
        ha="center",
        va="bottom"
    )

plt.ylim(0, 1.05)

plt.ylabel("Accuracy")
plt.xlabel("Machine Learning Models")
plt.title("Performance Comparison of Machine Learning Models")

plt.show()

import matplotlib.pyplot as plt

# Model names
models = ["Logistic Regression", "SVM", "Random Forest"]

# Accuracy values
accuracies = [accuracy, svm_accuracy, rf_accuracy]

plt.figure(figsize=(8,5))

bars = plt.bar(models, accuracies)

# Display accuracy values on each bar
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.01,
        f"{height:.2f}",
        ha="center"
    )

plt.ylim(0, 1.1)
plt.ylabel("Accuracy")
plt.title("Comparison of Machine Learning Models")
plt.show()

import pandas as pd

# Create a summary table
results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        svm_accuracy,
        rf_accuracy
    ]
})

print(results)

# Save to CSV
results.to_csv("Model_Comparison_Results.csv", index=False)

print("\nResults saved successfully!")