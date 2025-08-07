import pandas as pd
import matplotlib.pyplot as plt
from transformers import BertForSequenceClassification, BertTokenizer
from transformers import TextClassificationPipeline
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import Trainer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# # 1) Load dataset
# df = pd.read_csv(r"dataset\cleaned_fakenews_dataset.csv")  

# # 2) Check label column
# label_col = "label"
# if label_col not in df.columns:
#     raise KeyError(f"'{label_col}' column not found. Available columns: {list(df.columns)}")

# # 3) Count 0 (fake) vs 1 (real)
# counts = df[label_col].value_counts().sort_index()
# labels = ["Fake (0)", "Real (1)"]
# values = [counts.get(0, 0), counts.get(1, 0)]

# # 4) Plot pie chart
# plt.figure(figsize=(6, 6))
# plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff6666', '#66b3ff'])
# plt.title("Fake vs Real Article Distribution")
# plt.tight_layout()
# plt.show()

model_path = "saved_models/bert_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

df = pd.read_csv(r"dataset\cleaned_fakenews_dataset.csv")  
dataset = Dataset.from_pandas(df)
 
val_data = Dataset.from_pandas(df[["preprocessed", "label"]])

# Tokenize
def tokenize(batch):
    return tokenizer(batch['preprocessed'], padding=True, truncation=True)

tokenized_val = val_data.map(tokenize, batched=True)
tokenized_val.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

trainer = Trainer(model=model)

predictions = trainer.predict(tokenized_val)
y_pred = predictions.predictions.argmax(axis=1)
y_true = predictions.label_ids

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
disp.plot(cmap="Blues", values_format='d')   
plt.title("Confusion Matrix from Saved BERT Model")
plt.show()