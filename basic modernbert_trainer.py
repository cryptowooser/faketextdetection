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
import re
from spellchecker import SpellChecker
import textstat
import json

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

def count_spelling_errors(text: str, spell_checker: SpellChecker) -> int:
    """
    Counts the number of spelling errors in a given text.
    """
    # Simple word tokenization, handles basic punctuation.
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return 0
    misspelled = spell_checker.unknown(words)
    return len(misspelled)

def get_text_stats(text: str) -> tuple[float, float, float]:
    """
    Calculates readability scores for a given text.
    Returns Flesch Reading Ease, Flesch-Kincaid Grade, and Gunning Fog Index.
    """
    ease = textstat.flesch_reading_ease(text)
    grade = textstat.flesch_kincaid_grade(text)
    fog = textstat.gunning_fog(text)
    return ease, grade, fog

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
        
    # Initialize Spell Checker once
    spell = SpellChecker()
        
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

    # C) Calculate Spelling Error Counts
    spell_errors_1 = [count_spelling_errors(text, spell) for text in tqdm(train_df['text_1'], desc="Spell-checking text 1")]
    spell_errors_2 = [count_spelling_errors(text, spell) for text in tqdm(train_df['text_2'], desc="Spell-checking text 2")]
    spell_diffs = [s1 - s2 for s1, s2 in zip(spell_errors_1, spell_errors_2)]

    # D) Calculate Readability Stats
    stats1 = [get_text_stats(text) for text in tqdm(train_df['text_1'], desc="Stat-checking text 1")]
    stats2 = [get_text_stats(text) for text in tqdm(train_df['text_2'], desc="Stat-checking text 2")]

    # Unzip the stats and calculate differences
    flesch_ease_1, flesch_grade_1, gunning_fog_1 = zip(*stats1)
    flesch_ease_2, flesch_grade_2, gunning_fog_2 = zip(*stats2)

    flesch_ease_diffs = [s1 - s2 for s1, s2 in zip(flesch_ease_1, flesch_ease_2)]
    flesch_grade_diffs = [s1 - s2 for s1, s2 in zip(flesch_grade_1, flesch_grade_2)]
    gunning_fog_diffs = [s1 - s2 for s1, s2 in zip(gunning_fog_1, gunning_fog_2)]

    # E) Combine Features
    combined_features = [
        np.hstack((emb_diff, np.array([mlm_diff, spell_diff, flesch_ease_diff, flesch_grade_diff, gunning_fog_diff])))
        for emb_diff, mlm_diff, spell_diff, flesch_ease_diff, flesch_grade_diff, gunning_fog_diff in zip(
            embedding_diffs, mlm_diffs, spell_diffs, flesch_ease_diffs, flesch_grade_diffs, gunning_fog_diffs
        )
    ]
    train_df['features'] = combined_features

    end_time = time.time()
    print(f"Time to calculate all training features: {end_time - start_time:.2f} seconds")

    # --- 3. Run 5-Fold Cross-Validation with LightGBM ---
    print("\n--- Running 5-Fold Cross-Validation with LightGBM ---")
    X_features = np.vstack(train_df['features'].tolist())
    y = (train_df['winner'] - 1).values # Convert labels to 0 and 1

    # Create feature names for clarity and to prevent warnings
    embedding_dim = embedding_diffs[0].shape[0]
    feature_names = [f'embed_{i}' for i in range(embedding_dim)] + [
        'mlm_diff', 'spell_diff', 
        'flesch_ease_diff', 'flesch_grade_diff', 'gunning_fog_diff'
    ]
    X = pd.DataFrame(X_features, columns=feature_names)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_scores = [] 
    wrong_predictions = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Use LightGBM Classifier
        lgbm_model = LGBMClassifier(random_state=42, verbose=-1)
        lgbm_model.fit(X_train, y_train)
        
        preds = lgbm_model.predict(X_val)
        probas = lgbm_model.predict_proba(X_val)
        score = accuracy_score(y_val, preds)
        oof_scores.append(score)
        print(f"Fold {fold+1} Accuracy: {score:.4f}")

        # Log misclassified samples
        misclassified_mask = y_val != preds
        misclassified_indices_in_val_set = np.where(misclassified_mask)[0]

        for i in misclassified_indices_in_val_set:
            original_idx = val_idx[i]
            record = train_df.iloc[original_idx]
            
            wrong_sample = {
                'id': int(record['id']),
                'fold': fold + 1,
                'true_winner': int(record['winner']),
                'predicted_winner': int(preds[i] + 1),
                'confidence_for_predicted': float(probas[i].max()),
                'confidence_for_class_0': float(probas[i][0]),
                'confidence_for_class_1': float(probas[i][1]),
                'text_1': record['text_1'],
                'text_2': record['text_2']
            }
            wrong_predictions.append(wrong_sample)

    print("-" * 20)
    print(f"Average CV Accuracy: {np.mean(oof_scores):.4f} (+/- {np.std(oof_scores):.4f})")
    print("-" * 20)

    # Save misclassified samples to a JSON file
    if wrong_predictions:
        print(f"\nSaving {len(wrong_predictions)} misclassified samples to wrong_predictions.json...")
        with open('wrong_predictions.json', 'w', encoding='utf-8') as f:
            json.dump(wrong_predictions, f, indent=4, ensure_ascii=False)
        print("Done.")

    # --- 4. Train Final Model on Full Dataset ---
    print("\n--- Training Final LightGBM Model on Full Dataset ---")
    final_model = LGBMClassifier(random_state=42, verbose=-1)
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

        # C) Calculate Spelling Error Counts for Test Data
        test_spell_errors_1 = [count_spelling_errors(text, spell) for text in tqdm(test_df['text_1'], desc="Spell-checking test text 1")]
        test_spell_errors_2 = [count_spelling_errors(text, spell) for text in tqdm(test_df['text_2'], desc="Spell-checking test text 2")]
        test_spell_diffs = [s1 - s2 for s1, s2 in zip(test_spell_errors_1, test_spell_errors_2)]

        # D) Calculate Readability Stats for Test Data
        test_stats1 = [get_text_stats(text) for text in tqdm(test_df['text_1'], desc="Stat-checking test text 1")]
        test_stats2 = [get_text_stats(text) for text in tqdm(test_df['text_2'], desc="Stat-checking test text 2")]

        # Unzip the stats and calculate differences
        test_flesch_ease_1, test_flesch_grade_1, test_gunning_fog_1 = zip(*test_stats1)
        test_flesch_ease_2, test_flesch_grade_2, test_gunning_fog_2 = zip(*test_stats2)

        test_flesch_ease_diffs = [s1 - s2 for s1, s2 in zip(test_flesch_ease_1, test_flesch_ease_2)]
        test_flesch_grade_diffs = [s1 - s2 for s1, s2 in zip(test_flesch_grade_1, test_flesch_grade_2)]
        test_gunning_fog_diffs = [s1 - s2 for s1, s2 in zip(test_gunning_fog_1, test_gunning_fog_2)]

        # E) Combine Features for Test Data
        test_combined_features = [
            np.hstack((emb_diff, np.array([mlm_diff, spell_diff, flesch_ease_diff, flesch_grade_diff, gunning_fog_diff])))
            for emb_diff, mlm_diff, spell_diff, flesch_ease_diff, flesch_grade_diff, gunning_fog_diff in zip(
                test_embedding_diffs, test_mlm_diffs, test_spell_diffs, 
                test_flesch_ease_diffs, test_flesch_grade_diffs, test_gunning_fog_diffs
            )
        ]
        X_test_features = np.vstack(test_combined_features)
        X_test = pd.DataFrame(X_test_features, columns=feature_names)

        end_time = time.time()
        print(f"Time to calculate test features: {end_time - start_time:.2f} seconds")

        # --- 6. Make Predictions ---
        print("\n--- Making Predictions on Test Data ---")
        predictions = final_model.predict(X_test)

        # --- 7. Generate Submission File ---
        print("\n--- Generating Submission File ---")
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'real_text_id': predictions + 1 # Convert predictions back to 1 and 2
        })
        submission_path = 'submission.csv'
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission file saved to {submission_path}")
        print(submission_df.head()) 

    # --- 8. Display Feature Importances ---
    print("\n--- Feature Importances ---")
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': final_model.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    # Filter out embedding features with 0 importance to avoid clutter
    mask = (feature_importance_df['feature'].str.startswith('embed_')) & (feature_importance_df['importance'] == 0)
    filtered_feature_importance_df = feature_importance_df[~mask].reset_index(drop=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(filtered_feature_importance_df) 