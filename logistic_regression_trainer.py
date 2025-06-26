import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# --- BERT Embedding Setup ---
MODEL_NAME = "answerdotai/ModernBERT-large"

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set precision for better performance on newer GPUs
if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device) # Move model to the selected device
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # Exit or handle gracefully if the model is essential
    exit()

def get_bert_embedding(text: str) -> np.ndarray:
    """
    Takes a text string and returns its [CLS] token embedding as a NumPy array.
    Handles potential tokenization issues with long texts.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    # Move inputs to the selected device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
    return cls_embedding

# --- Data Loading ---
def load_and_process_data(csv_path='data/train.csv', articles_path='data/train'):
    """
    Loads the training data, processes the text files, and returns a pandas DataFrame.
    """
    if not os.path.exists(csv_path):
        print(f"Error: The file at {csv_path} was not found.")
        return None
    train_df = pd.read_csv(csv_path)

    data = []
    for _, row in train_df.iterrows():
        article_id = row['id']
        real_text_id = row['real_text_id']
        article_folder = f"article_{article_id:04d}"
        folder_path = os.path.join(articles_path, article_folder)
        
        try:
            with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f:
                text1 = f.read()
            with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f:
                text2 = f.read()
        except FileNotFoundError:
            continue
            
        data.append({
            'id': article_id,
            'text_1': text1,
            'text_2': text2,
            'winner': real_text_id
        })
        
    return pd.DataFrame(data)

def load_test_data(articles_path='data/test'):
    """
    Loads the test data by scanning subfolders, processes the text files, 
    and returns a pandas DataFrame.
    """
    if not os.path.isdir(articles_path):
        print(f"Error: The directory at {articles_path} was not found.")
        return None

    data = []
    # List subdirectories in the articles_path
    try:
        subfolders = [f for f in os.listdir(articles_path) if os.path.isdir(os.path.join(articles_path, f))]
    except FileNotFoundError:
        print(f"Error: The directory at {articles_path} was not found.")
        return None


    for folder_name in subfolders:
        try:
            # e.g., 'article_1501' -> 1501
            article_id = int(folder_name.split('_')[1])
        except (ValueError, IndexError):
            print(f"Warning: Could not parse article ID from folder name {folder_name}. Skipping.")
            continue
        
        folder_path = os.path.join(articles_path, folder_name)
        
        try:
            with open(os.path.join(folder_path, 'file_1.txt'), 'r', encoding='utf-8') as f:
                text1 = f.read()
            with open(os.path.join(folder_path, 'file_2.txt'), 'r', encoding='utf-8') as f:
                text2 = f.read()
        except FileNotFoundError:
            print(f"Warning: Text files not found in {folder_path} for test id {article_id}. Skipping.")
            continue
            
        data.append({
            'id': article_id,
            'text_1': text1,
            'text_2': text2,
        })
        
    return pd.DataFrame(data)

# --- Main Execution ---
if __name__ == '__main__':
    # --- 1. Load and Process Training Data ---
    print("--- Loading and Processing Training Data ---")
    train_df = load_and_process_data()

    if train_df is None or train_df.empty:
        print("Could not load training data. Exiting.")
        exit()
        
    # Calculate embeddings for training data
    print("Calculating embeddings for training data...")
    start_time = time.time()
    train_df['text_1_embedding'] = train_df['text_1'].apply(get_bert_embedding)
    train_df['text_2_embedding'] = train_df['text_2'].apply(get_bert_embedding)
    train_df['embedding_diff'] = train_df['text_1_embedding'] - train_df['text_2_embedding']
    end_time = time.time()
    print(f"Time to calculate training embeddings: {end_time - start_time:.2f} seconds")

    # --- 2. Train the Model on the Full Training Dataset ---
    print("\n--- Training Logistic Regression Model on Full Dataset ---")
    X_train = np.vstack(train_df['embedding_diff'].values)
    y_train = train_df['winner'].values

    lr_model = LogisticRegression(random_state=42)
    start_time = time.time()
    lr_model.fit(X_train, y_train)
    end_time = time.time()
    print(f"Time to train logistic regression: {end_time - start_time:.4f} seconds")

    # --- 3. Load and Process Test Data ---
    print("\n--- Loading and Processing Test Data ---")
    test_df = load_test_data()

    if test_df is not None and not test_df.empty:
        # Calculate embeddings for test data
        print("Calculating embeddings for test data...")
        start_time = time.time()
        test_df['text_1_embedding'] = test_df['text_1'].apply(get_bert_embedding)
        test_df['text_2_embedding'] = test_df['text_2'].apply(get_bert_embedding)
        test_df['embedding_diff'] = test_df['text_1_embedding'] - test_df['text_2_embedding']
        end_time = time.time()
        print(f"Time to calculate test embeddings: {end_time - start_time:.2f} seconds")

        # --- 4. Make Predictions ---
        print("\n--- Making Predictions on Test Data ---")
        X_test = np.vstack(test_df['embedding_diff'].values)
        predictions = lr_model.predict(X_test)

        # --- 5. Generate Submission File ---
        print("\n--- Generating Submission File ---")
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'real_text_id': predictions
        })

        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file saved to {submission_path}")
        print(submission_df.head()) 