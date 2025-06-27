import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time

# --- BERT Embedding Setup ---
MODEL_NAME = "answerdotai/ModernBERT-large"
BATCH_SIZE = 32

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

def get_bert_embeddings_in_batches(texts: list[str], batch_size: int) -> list[np.ndarray]:
    """
    Takes a list of text strings and returns their [CLS] token embeddings as a list of NumPy arrays,
    processed in batches for efficiency.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", max_length=512, truncation=True, padding=True)
        # Move inputs to the selected device
        inputs = {key: val.to(device) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get the CLS token embedding for each item in the batch
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.extend([emb for emb in cls_embeddings])
        
    return all_embeddings

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
    print("Calculating embeddings for training data (in batches)...")
    start_time = time.time()

    # Create a single list of all texts to be processed
    all_train_texts = train_df['text_1'].tolist() + train_df['text_2'].tolist()
    all_train_embeddings = get_bert_embeddings_in_batches(all_train_texts, batch_size=BATCH_SIZE)

    # Split the combined embeddings back into text_1 and text_2
    num_samples = len(train_df)
    train_embeddings_1 = all_train_embeddings[:num_samples]
    train_embeddings_2 = all_train_embeddings[num_samples:]

    # Now create the difference vector
    train_df['embedding_diff'] = [e1 - e2 for e1, e2 in zip(train_embeddings_1, train_embeddings_2)]
    
    end_time = time.time()
    print(f"Time to calculate training embeddings: {end_time - start_time:.2f} seconds")

    # --- 2. Run 5-Fold Cross-Validation ---
    print("\n--- Running 5-Fold Cross-Validation ---")
    X = np.vstack(train_df['embedding_diff'].values)
    y = train_df['winner'].values

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_scores = [] 

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train, y_train)
        
        preds = lr_model.predict(X_val)
        score = accuracy_score(y_val, preds)
        oof_scores.append(score)
        print(f"Fold {fold+1} Accuracy: {score:.4f}")

    print("-" * 20)
    print(f"Average CV Accuracy: {np.mean(oof_scores):.4f} (+/- {np.std(oof_scores):.4f})")
    print("-" * 20)

    # --- 3. Train Final Model on Full Dataset ---
    print("\n--- Training Final Model on Full Dataset ---")
    final_model = LogisticRegression(random_state=42)
    start_time = time.time()
    final_model.fit(X, y)
    end_time = time.time()
    print(f"Time to train final model: {end_time - start_time:.4f} seconds")


    # --- 4. Load and Process Test Data ---
    print("\n--- Loading and Processing Test Data ---")
    test_df = load_test_data()

    if test_df is not None and not test_df.empty:
        # Calculate embeddings for test data
        print("Calculating embeddings for test data (in batches)...")
        start_time = time.time()

        all_test_texts = test_df['text_1'].tolist() + test_df['text_2'].tolist()
        all_test_embeddings = get_bert_embeddings_in_batches(all_test_texts, batch_size=BATCH_SIZE)
        
        num_test_samples = len(test_df)
        test_embeddings_1 = all_test_embeddings[:num_test_samples]
        test_embeddings_2 = all_test_embeddings[num_test_samples:]
        
        test_df['embedding_diff'] = [e1 - e2 for e1, e2 in zip(test_embeddings_1, test_embeddings_2)]

        end_time = time.time()
        print(f"Time to calculate test embeddings: {end_time - start_time:.2f} seconds")

        # --- 5. Make Predictions ---
        print("\n--- Making Predictions on Test Data ---")
        X_test = np.vstack(test_df['embedding_diff'].values)
        predictions = final_model.predict(X_test)

        # --- 6. Generate Submission File ---
        print("\n--- Generating Submission File ---")
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'real_text_id': predictions
        })

        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file saved to {submission_path}")
        print(submission_df.head()) 