from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import VotingClassifier

df = pd.read_csv(r"dataset\cleaned_fakenews_dataset.csv")  
dataset_folder = os.path.join(os.getcwd(), 'dataset')
file_path = os.path.join(dataset_folder, 'tfidf_vectorizer.pkl')

with open(file_path, "rb") as f:
    vectorizer = pickle.load(f)

X = vectorizer.fit_transform(df['preprocessed'])  # convert to TF-IDF matrix
y = df['label']  # target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
# Evaluate
print("Accuracy (Naive Bayes):", accuracy_score(y_test, predictions))
print("\nClassification Report (Naive Bayes):\n", classification_report(y_test, predictions))

# Logistic Regression model 

# Train Logistic Regression model
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

# Predict
predictions_lr = model_lr.predict(X_test)

# Evaluate
print("Accuracy (Logistic Regression):", accuracy_score(y_test, predictions_lr))
print("\nClassification Report (Logistic Regression):\n", classification_report(y_test, predictions_lr))

# Define the models
nb_model = MultinomialNB()
lr_model = LogisticRegression(max_iter=1000)

# Combine models into a VotingClassifier
voting_clf = VotingClassifier(
    estimators=[('nb', nb_model), ('lr', lr_model)],
    voting='soft'  # or 'hard'
)

# Fit on TF-IDF features
voting_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = voting_clf.predict(X_test)
print("Accuracy (VotingClassifier):", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

os.makedirs("saved_models", exist_ok=True)

# Save Naive Bayes model
with open("saved_models/naive_bayes_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save Logistic Regression model
with open("saved_models/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model_lr, f)

# Save Voting Classifier model
with open("saved_models/voting_classifier_model.pkl", "wb") as f:
    pickle.dump(voting_clf, f)

print("All models saved successfully.")