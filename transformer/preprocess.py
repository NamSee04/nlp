import torch
from datasets import load_dataset
from tqdm import tqdm
import os
import pickle
from utils import preprocess_text, build_vocabulary

def preprocess_dataset(direction='en-vi', min_freq=2, cache_dir='cache'):
    """
    Preprocess the dataset and build vocabularies
    
    Args:
        direction: 'en-vi' for English to Vietnamese, 'vi-en' for Vietnamese to English
        min_freq: minimum frequency for words to be included in vocabulary
        cache_dir: directory to cache processed data
        
    Returns:
        train_data, val_data, test_data, src_vocab, tgt_vocab
    """
    # Set source and target languages based on direction
    if direction == 'en-vi':
        src_lang, tgt_lang = 'English', 'Vietnamese'
    elif direction == 'vi-en':
        src_lang, tgt_lang = 'Vietnamese', 'English'
    else:
        raise ValueError("direction must be either 'en-vi' or 'vi-en'")
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    vocab_cache_file = f'{cache_dir}/{src_lang}_{tgt_lang}_vocab.pkl'
    data_cache_file = f'{cache_dir}/{src_lang}_{tgt_lang}_data.pkl'
    
    # Check if preprocessed data exists
    if os.path.exists(data_cache_file) and os.path.exists(vocab_cache_file):
        print("Loading preprocessed data from cache...")
        with open(data_cache_file, 'rb') as f:
            train_data, val_data, test_data = pickle.load(f)
        with open(vocab_cache_file, 'rb') as f:
            src_vocab, tgt_vocab = pickle.load(f)
        return train_data, val_data, test_data, src_vocab, tgt_vocab
    
    print("Preprocessing data...")
    
    # Load dataset with parameters to work around caching issues
    dataset = load_dataset('harouzie/vi_en-translation', download_mode='force_redownload')
    
    # Create train/val/test split if not already done
    if 'validation' not in dataset or 'test' not in dataset:
        splits = dataset['train'].train_test_split(test_size=0.2)
        train_data = splits['train']
        temp_splits = splits['test'].train_test_split(test_size=0.5)
        val_data = temp_splits['train']
        test_data = temp_splits['test']
    else:
        train_data = dataset['train']
        val_data = dataset['validation']
        test_data = dataset['test']
    
    # Preprocess the data
    print("Preprocessing text data...")
    
    # Extract source and target texts
    src_texts = [preprocess_text(item[src_lang], src_lang) for item in tqdm(train_data, desc="Processing source texts")]
    tgt_texts = [preprocess_text(item[tgt_lang], tgt_lang) for item in tqdm(train_data, desc="Processing target texts")]
    
    # Build vocabularies
    print("Building vocabularies...")
    src_vocab = build_vocabulary(src_texts, min_freq=min_freq)
    tgt_vocab = build_vocabulary(tgt_texts, min_freq=min_freq)
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Save vocabularies and data
    with open(vocab_cache_file, 'wb') as f:
        pickle.dump((src_vocab, tgt_vocab), f)
    
    with open(data_cache_file, 'wb') as f:
        pickle.dump((train_data, val_data, test_data), f)
    
    return train_data, val_data, test_data, src_vocab, tgt_vocab

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess dataset and build vocabularies")
    parser.add_argument("--direction", type=str, default="en-vi", choices=["en-vi", "vi-en"], 
                        help="Translation direction")
    parser.add_argument("--min_freq", type=int, default=2, 
                        help="Minimum frequency for words to be included in vocabulary")
    parser.add_argument("--cache_dir", type=str, default="cache", 
                        help="Directory to cache processed data")
    
    args = parser.parse_args()
    
    preprocess_dataset(args.direction, args.min_freq, args.cache_dir)
