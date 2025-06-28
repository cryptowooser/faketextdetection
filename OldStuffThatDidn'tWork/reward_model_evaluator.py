import pandas as pd
import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# --- Reward Model Setup ---
MODEL_NAME = "LxzGordon/URM-LLaMa-3.1-8B"
BATCH_SIZE = 2 # This new method uses more memory per item, so batch size is reduced.

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set precision for better performance on newer GPUs
if device.type == 'cuda':
    torch.set_float32_matmul_precision('high')

# --- Memory Optimization: 8-bit quantization ---
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Set pad token if it's not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.float16, # Align with the model's custom code
        trust_remote_code=True, # Required by this model
        device_map="auto" # Automatically handle device placement
    )
    # Important: also update model config
    model.config.pad_token_id = tokenizer.pad_token_id

    # model.to(device) # No need, device_map handles it
    model.eval()
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit()

PROMPT_TEMPLATE = """Here are two pieces of text, one of which was corrupted. Please respond with "Text 1" if Text 1 is the corrupted text, and "Text 2" if Text 2 is the corrupted text.

Text 1:
{text_1}

Text 2:
{text_2}
"""

def get_preference_scores(batch_text_pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
    """
    Takes a batch of (text1, text2) pairs and returns reward scores for two possible
    answers: "Text 1" and "Text 2", based on which is the corrupted one.
    """
    choice1 = "Text 1"
    choice2 = "Text 2"

    # For each pair, we create two conversations: one for each choice.
    conversations_for_scoring = []
    for text1, text2 in batch_text_pairs:
        # Create the user prompt with the two texts
        user_prompt = PROMPT_TEMPLATE.format(text_1=text1, text_2=text2)
        
        # Create conversation for Choice 1
        messages1 = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": choice1}]
        conversations_for_scoring.append(tokenizer.apply_chat_template(messages1, tokenize=False))

        # Create conversation for Choice 2
        messages2 = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": choice2}]
        conversations_for_scoring.append(tokenizer.apply_chat_template(messages2, tokenize=False))

    # Tokenize the entire batch of conversations (2 conversations per original pair)
    inputs = tokenizer(
        conversations_for_scoring,
        return_tensors="pt",
        max_length=4096, # Increased max_length to handle two full texts
        truncation=True,
        padding=True,
    )
    
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        scores = logits[:, 0].cpu().numpy().flatten().tolist()
    
    # Unpack scores back into pairs
    score_pairs = []
    for i in range(0, len(scores), 2):
        score_pairs.append((scores[i], scores[i+1]))
        
    return score_pairs

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
            'winner': real_text_id # winner = real text
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

    # --- 2. Get Preference Scores and Make Predictions ---
    print("\n--- Getting preference scores for all text pairs ---")
    
    all_texts_1 = train_df['text_1'].tolist()
    all_texts_2 = train_df['text_2'].tolist()
    text_pairs = list(zip(all_texts_1, all_texts_2))
    
    # --- Debug: Print the first prompt ---
    if text_pairs:
        print("\n--- Sample Prompt for Debugging ---")
        first_text1, first_text2 = text_pairs[0]
        debug_prompt = PROMPT_TEMPLATE.format(text_1=first_text1, text_2=first_text2)
        print(debug_prompt)
        print("--- End of Sample Prompt ---\n")
    
    all_score_pairs = []
    for i in tqdm(range(0, len(text_pairs), BATCH_SIZE), desc="Scoring Choice Batches"):
        batch = text_pairs[i:i+BATCH_SIZE]
        score_pairs = get_preference_scores(batch)
        all_score_pairs.extend(score_pairs)

    # The prompt asks the model to identify the *corrupted* text.
    # Our 'winner' column identifies the *real* text. So, the corrupted text is the other one.
    actual_corrupted = [2 if w == 1 else 1 for w in train_df['winner']]
    
    predictions_corrupted = []
    scores_for_choice1 = []
    scores_for_choice2 = []

    for score1, score2 in all_score_pairs:
        if score1 > score2:
            # Model believes "Text 1" is the corrupted one
            predictions_corrupted.append(1)
        else:
            # Model believes "Text 2" is the corrupted one
            predictions_corrupted.append(2)
        scores_for_choice1.append(score1)
        scores_for_choice2.append(score2)

    # --- 4. Evaluate Predictions ---
    print("\n--- Evaluating predictions ---")
    
    f1_micro = f1_score(actual_corrupted, predictions_corrupted, average='micro')
    f1_macro = f1_score(actual_corrupted, predictions_corrupted, average='macro')
    accuracy = accuracy_score(actual_corrupted, predictions_corrupted)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (Micro): {f1_micro:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")

    # --- 5. Save results ---
    results_df = train_df.copy()
    results_df['score_for_choice1'] = scores_for_choice1
    results_df['score_for_choice2'] = scores_for_choice2
    results_df['predicted_corrupted_text'] = predictions_corrupted
    results_df['actual_corrupted_text'] = actual_corrupted
    
    output_path = 'reward_model_choice_predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved predictions and scores to {output_path}")
    print(results_df.head()) 