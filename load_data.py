import pandas as pd
import os

def load_and_process_data(csv_path='data/train.csv', articles_path='data/train'):
    """
    Loads the training data, processes the text files, and returns a pandas DataFrame.

    Args:
        csv_path (str): The path to the train.csv file.
        articles_path (str): The base path to the article folders.

    Returns:
        pd.DataFrame: A DataFrame with columns 'id', 'real_text', and 'fake_text'.
    """
    try:
        train_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} was not found.")
        return None

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
            print(f"Warning: Text files not found in {folder_path}. Skipping.")
            continue
            
        if real_text_id == 1:
            real_text = text1
            fake_text = text2
        else:
            real_text = text2
            fake_text = text1
            
        data.append({
            'id': article_id,
            'real_text': real_text,
            'fake_text': fake_text
        })
        
    processed_df = pd.DataFrame(data)
    return processed_df

if __name__ == '__main__':
    processed_data = load_and_process_data()
    if processed_data is not None:
        print(processed_data.head())
        
        # Save the first 10 rows to a JSON file
        output_path = 'data_head.json'
        try:
            processed_data.head(10).to_json(output_path, orient='records', indent=4)
            print(f"Successfully saved the first 10 rows to {output_path}")
        except Exception as e:
            print(f"Error saving to JSON: {e}") 