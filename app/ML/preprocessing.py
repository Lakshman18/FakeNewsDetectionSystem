import pandas as pd
import re 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import numpy as np

def arrangeDataset():
    # Load datasets
    true_df_ds1 = pd.read_csv(r"dataset\True.csv")
    false_df_ds1 = pd.read_csv(r"dataset\Fake.csv")

    # Add labels
    true_df_ds1['label'] = 1   # real
    false_df_ds1['label'] = 0  # fake

    # Combine datasets
    df = pd.concat([true_df_ds1, false_df_ds1], ignore_index=True)

    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Combine title and text label 
    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'] + " " + df['text']
    elif 'text' in df.columns:
        df['content'] = df['text']
    else:
        raise Exception("Dataset must contain 'title' or 'text' columns")
    
    # Apply cleaning
    df['cleaned'] = df['content'].apply(clean_text)

    # Apply to a DataFrame column
    df['preprocessed'] = df['cleaned'].apply(preprocess_text)
    df = df[df['preprocessed'].notnull() & (df['preprocessed'].str.strip() != '')]

    # Separate and shuffle real and fake entries
    df_real = df[df['label'] == 1].sample(frac=1, random_state=42).reset_index(drop=True)
    df_fake = df[df['label'] == 0].sample(frac=1, random_state=42).reset_index(drop=True)

    # Select exactly 30k from each
    df_real_21k = df_real.iloc[:21000]
    df_fake_21k = df_fake.iloc[:21000]

    # Interleave real and fake: real, fake, real, fake, ...
    interleaved_df = pd.DataFrame()
    interleaved_df = pd.concat([df_real_21k, df_fake_21k], ignore_index=True)
    interleaved_df = interleaved_df.iloc[np.ravel(np.column_stack((range(21000), range(21000, 42000))))].reset_index(drop=True)

    # Confirm alternating labels
    df = interleaved_df

    vectorizer = TfidfVectorizer(
        max_features=10000,  # or try 5000, 15000
        ngram_range=(1,2),   # include bigrams
        stop_words='english',  # removes common stop words
        sublinear_tf=True     # more emphasis on rare terms
    )

    # vectorizer = TfidfVectorizer(
    #     max_features=5000,         # keep top 5000 words
    #     stop_words='english',      # remove common stopwords
    #     ngram_range=(1, 3)         # use unigrams + bigrams
    # )

    # Save the data set in a new file
    df[['preprocessed', 'label']].to_csv(r"dataset\cleaned_fakenews_dataset.csv", index=False)
    X = vectorizer.fit_transform(df['preprocessed']) 

    # save the vectorizer in a file
    dataset_folder = os.path.join(os.getcwd(), 'dataset')
    os.makedirs(dataset_folder, exist_ok=True)  # Create the dataset folder if it doesn't exist

    file_path = os.path.join(dataset_folder, 'tfidf_vectorizer.pkl')

    with open(file_path, "wb") as f:
        pickle.dump(vectorizer, f)


# text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)                 # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)              # Remove punctuation & digits
    text = re.sub(r'\s+', ' ', text).strip()          # Remove extra spaces
    return text

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# preprocessor
def preprocess_text(text):
    try:
        
        text = clean_text(text)
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error processing text: {text}\n{e}")
        return ""

# arrangeDataset()




