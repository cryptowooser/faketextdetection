import torch
from transformers import AutoTokenizer, AutoModel

# Load the model and tokenizer (do this once)
MODEL_NAME = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def get_bert_embedding(text: str) -> torch.Tensor:
    """
    Takes a text string and returns its [CLS] token embedding as a 1D tensor.
    """
    # It's good practice to handle text length
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    with torch.no_grad(): # We don't need to calculate gradients
        outputs = model(**inputs)
    
    # The [CLS] token embedding is the first token of the last hidden state
    cls_embedding = outputs.last_hidden_state[0, 0, :]
    return cls_embedding

if __name__ == "__main__":
    sentence = "What's a pineapple?"
    embedding = get_bert_embedding(sentence)
    print(f"The embedding for the sentence: '{sentence}' is:")
    print(embedding)
    print(f"The shape of the embedding is: {embedding.shape}")