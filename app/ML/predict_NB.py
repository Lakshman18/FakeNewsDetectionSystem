import pickle
import matplotlib.pyplot as plt
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from app.ML.preprocessing import preprocess_text
import os  

def analyze_text_with_lime_NB(text):
    BASE_DIR = os.getcwd()  # D:\herts\Sem C\FakeNewsDetectionSystem\FakeNewsDetectionSystem

    model_path = os.path.join(BASE_DIR, "app", "ML", "saved_models", "naive_bayes_model.pkl")
    vectorizer_path = os.path.join(BASE_DIR,  "app", "ML", "dataset", "tfidf_vectorizer.pkl")
    # --------- Load model and vectorizer ---------
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    # ----------- Preprocess -----------
    preprocessed_text = preprocess_text(text)
    X_input = vectorizer.transform([preprocessed_text])
    prediction = model.predict(X_input)[0]
    prediction_label = "Real" if prediction == 1 else "Fake"
    confidence = max(model.predict_proba(X_input)[0]) * 100

    # ----------- LIME Explanation -----------
    pipeline_model = make_pipeline(vectorizer, model)
    class_names = ["Fake", "Real"]
    explainer = LimeTextExplainer(class_names=class_names)

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=pipeline_model.predict_proba,
        num_features=10
    )

    # Save HTML
    output_path = os.path.join(BASE_DIR, "app", "ML", "lime_explanation.html")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(exp.as_html())


    # Return result as a dictionary
    return {
        "prediction": prediction_label,
        "confidence": confidence,
        "top_features": exp.as_list(),
    }

 
