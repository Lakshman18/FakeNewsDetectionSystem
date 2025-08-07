import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np
from app.ML.preprocessing import preprocess_text  # Optional

def analyze_text_with_lime_Bert(text):
    BASE_DIR = os.getcwd()
    model_path = os.path.join(BASE_DIR, "app", "ML", "saved_models", "bert_model")

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Custom LIME-compatible prediction function
    def predict_proba(texts):
        batch_size = 16  # fixed batch size inside the function
        probs = []
        model.eval()

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                batch_probs = torch.nn.functional.softmax(logits, dim=1)
                probs.append(batch_probs.cpu())

        return torch.cat(probs, dim=0).numpy()


    # Predict for the given single text
    # ----------- Preprocess -----------
    text = preprocess_text(text)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

    labels_map = {0: "Fake", 1: "Real"}
    prediction_label = labels_map[predicted_class.item()]
    confidence_score = confidence.item() * 100
    print(prediction_label)
    # LIME explanation
    class_names = ["Fake", "Real"]
    explainer = LimeTextExplainer(class_names=class_names, bow=True)
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=10,
        num_samples=100
    )
    # Save HTML
    output_path = os.path.join(BASE_DIR, "app", "ML", "lime_explanation.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(exp.as_html())

    return {
        "prediction": prediction_label,
        "confidence": confidence_score,
        "top_features": exp.as_list(),
    }
