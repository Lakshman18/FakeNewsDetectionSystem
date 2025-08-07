from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"dataset\cleaned_fakenews_dataset.csv")  
dataset_folder = os.path.join(os.getcwd(), 'dataset')
file_path = os.path.join(dataset_folder, 'tfidf_vectorizer.pkl')

with open(file_path, "rb") as f:
    vectorizer = pickle.load(f)

# Feature matrix and target
X = vectorizer.transform(df['preprocessed'])  # Use only transform
y = df['label']

# Setup 5-Fold Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
accuracies = []
precision_scores = []
recall_scores = []
f1_scores = []
all_true = []
all_preds = []

for train_index, test_index in skf.split(X, y):
    print(f"\n--- Fold {fold} ---")

    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_val)

    # Evaluation
    acc = accuracy_score(y_val, predictions)
    report_dict = classification_report(y_val, predictions, output_dict=True)

    # Macro average scores 
    precision_scores.append(report_dict["macro avg"]["precision"])
    recall_scores.append(report_dict["macro avg"]["recall"])
    f1_scores.append(report_dict["macro avg"]["f1-score"])
    accuracies.append(acc)

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_val, predictions, digits=4))
    all_true.extend(y_val)
    all_preds.extend(predictions)
    fold += 1

# Final averaged metrics
print("\n==============================")
print(f"Average Accuracy: {np.mean(accuracies):.4f}")
print(f"Average Precision (macro avg): {np.mean(precision_scores):.4f}")
print(f"Average Recall (macro avg): {np.mean(recall_scores):.4f}")
print(f"Average F1-score (macro avg): {np.mean(f1_scores):.4f}")

# Save logistic_regression_model
os.makedirs("saved_models", exist_ok=True)
with open("saved_models/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Overall confusion matrix
cm_total = confusion_matrix(all_true, all_preds)
print("\n=== Overall Confusion Matrix ===")
print(cm_total)

# visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm_total, display_labels=model.classes_)
fig, ax = plt.subplots(figsize=(6, 6))   
disp.plot(cmap='Blues', ax=ax, values_format='d')   
plt.title("Overall Confusion Matrix (Logistic Regression)")
plt.tight_layout()
plt.show()