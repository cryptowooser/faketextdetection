import pandas as pd
import os
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from tqdm import tqdm

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
    # CRITICAL CHANGE: Load the model with the MLM head for predictions
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME) 
    model.to(device)
    model.eval() # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
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

def get_mlm_scores_fast(texts: list[str], model, tokenizer, device, batch_size: int) -> list[float]:
    """
    Calculates a pseudo-log-likelihood score for each text in a list, processed in batches.
    This is a MUCH faster method that requires only a single forward pass per batch.
    """
    all_scores = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Fast MLM Scoring Batches"):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(batch_texts, return_tensors="pt", max_length=512, truncation=True, padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        
        with torch.no_grad():
            # Perform a single forward pass on the unmasked text
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Shift logits and labels to align them for calculating the loss
        # We want to predict token `i` using the output at position `i-1`
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Use log_softmax to get log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather the log probabilities of the true tokens
        # Flatten the tensors to easily calculate the loss for the whole batch
        true_token_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # We need to ignore the padding tokens in our calculation
        # The attention mask for the shifted labels tells us which tokens are not padding
        valid_token_mask = (shift_labels != tokenizer.pad_token_id)
        
        # Sum the log probabilities for each sequence in the batch, ignoring padding
        # We multiply by the mask (which is 0 for padding, 1 for real tokens)
        summed_log_probs = (true_token_log_probs * valid_token_mask).sum(dim=1)
        
        # Count the number of actual tokens in each sequence (again, ignoring padding)
        num_tokens = valid_token_mask.sum(dim=1)
        
        # Avoid division by zero for empty/fully-padded sequences
        num_tokens[num_tokens == 0] = 1
        
        # Calculate the average log-likelihood for each sequence
        scores = (summed_log_probs / num_tokens).cpu().numpy().tolist()
        
        all_scores.extend(scores)
        
    return all_scores

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
        
    # --- 2. Calculate Features for Training Data ---
    print("Calculating features for training data. This may take a while...")
    start_time = time.time()

    # A) Calculate BERT Embeddings
    all_train_texts = train_df['text_1'].tolist() + train_df['text_2'].tolist()
    # Note: get_bert_embeddings_in_batches needs to be adapted to use the base model
    # Let's define a separate function for base model embeddings to keep things clean.
    def get_base_model_embeddings(texts, model, tokenizer, device, batch_size):
        all_embeddings = []
        base_model = model.base_model # Access the underlying DeBERTa model without the MLM head
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding Batches"):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", max_length=512, truncation=True, padding=True)
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                outputs = base_model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.extend([emb for emb in cls_embeddings])
        return all_embeddings

    all_train_embeddings = get_base_model_embeddings(all_train_texts, model, tokenizer, device, BATCH_SIZE)
    num_samples = len(train_df)
    train_embeddings_1 = all_train_embeddings[:num_samples]
    train_embeddings_2 = all_train_embeddings[num_samples:]
    embedding_diffs = [e1 - e2 for e1, e2 in zip(train_embeddings_1, train_embeddings_2)]

    # B) Calculate MLM Coherence Scores
    mlm_scores_1 = get_mlm_scores_fast(train_df['text_1'].tolist(), model, tokenizer, device, BATCH_SIZE)
    mlm_scores_2 = get_mlm_scores_fast(train_df['text_2'].tolist(), model, tokenizer, device, BATCH_SIZE)
    mlm_diffs = [s1 - s2 for s1, s2 in zip(mlm_scores_1, mlm_scores_2)]

    # C) Combine Features
    combined_features = [np.hstack((emb_diff, np.array([mlm_diff]))) for emb_diff, mlm_diff in zip(embedding_diffs, mlm_diffs)]
    train_df['features'] = combined_features

    end_time = time.time()
    print(f"Time to calculate all training features: {end_time - start_time:.2f} seconds")

    # --- 3. Run 5-Fold Cross-Validation with LightGBM ---
    print("\n--- Running 5-Fold Cross-Validation with LightGBM ---")
    X = np.vstack(train_df['features'].tolist())
    y = train_df['winner'].values

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_scores = [] 

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Use LightGBM Classifier
        lgbm_model = LGBMClassifier(random_state=42)
        lgbm_model.fit(X_train, y_train)
        
        preds = lgbm_model.predict(X_val)
        score = accuracy_score(y_val, preds)
        oof_scores.append(score)
        print(f"Fold {fold+1} Accuracy: {score:.4f}")

    print("-" * 20)
    print(f"Average CV Accuracy: {np.mean(oof_scores):.4f} (+/- {np.std(oof_scores):.4f})")
    print("-" * 20)

    # --- 4. Train Final Model on Full Dataset ---
    print("\n--- Training Final LightGBM Model on Full Dataset ---")
    final_model = LGBMClassifier(random_state=42)
    final_model.fit(X, y)

    # --- 5. Load and Process Test Data ---
    print("\n--- Loading and Processing Test Data ---")
    test_df = load_test_data()

    if test_df is not None and not test_df.empty:
        print("Calculating features for test data. This may take a while...")
        start_time = time.time()
        
        # A) Calculate BERT Embeddings for Test Data
        all_test_texts = test_df['text_1'].tolist() + test_df['text_2'].tolist()
        all_test_embeddings = get_base_model_embeddings(all_test_texts, model, tokenizer, device, BATCH_SIZE)
        num_test_samples = len(test_df)
        test_embeddings_1 = all_test_embeddings[:num_test_samples]
        test_embeddings_2 = all_test_embeddings[num_test_samples:]
        test_embedding_diffs = [e1 - e2 for e1, e2 in zip(test_embeddings_1, test_embeddings_2)]

        # B) Calculate MLM Coherence Scores for Test Data
        test_mlm_scores_1 = get_mlm_scores_fast(test_df['text_1'].tolist(), model, tokenizer, device, BATCH_SIZE)
        test_mlm_scores_2 = get_mlm_scores_fast(test_df['text_2'].tolist(), model, tokenizer, device, BATCH_SIZE)
        test_mlm_diffs = [s1 - s2 for s1, s2 in zip(test_mlm_scores_1, test_mlm_scores_2)]

        # C) Combine Features for Test Data
        test_combined_features = [np.hstack((emb_diff, np.array([mlm_diff]))) for emb_diff, mlm_diff in zip(test_embedding_diffs, test_mlm_diffs)]
        X_test = np.vstack(test_combined_features)

        end_time = time.time()
        print(f"Time to calculate test features: {end_time - start_time:.2f} seconds")

        # --- 6. Make Predictions ---
        print("\n--- Making Predictions on Test Data ---")
        predictions = final_model.predict(X_test)

        # --- 7. Generate Submission File ---
        print("\n--- Generating Submission File ---")
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'real_text_id': predictions
        })
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file saved to {submission_path}")
        print(submission_df.head()) 