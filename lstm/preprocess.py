import torch
from datasets import load_dataset
from tqdm import tqdm
import os
import pickle
import re
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from utils import preprocess_text

def train_subword_tokenizer(texts, vocab_size=8000, special_tokens=None):
    """
    Train a BPE tokenizer on the provided texts
    
    Args:
        texts: List of preprocessed text samples
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens to include
        
    Returns:
        Trained tokenizer
    """
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(texts, trainer)
    return tokenizer

def filter_by_length(src_text, tgt_text, max_len=50):
    """Filter out sentence pairs that exceed maximum length"""
    if len(src_text.split()) > max_len or len(tgt_text.split()) > max_len:
        return False
    return True

def filter_by_ratio(src_text, tgt_text, max_ratio=2.0):
    """Filter out sentence pairs with extreme length ratios"""
    src_len = len(src_text.split())
    tgt_len = len(tgt_text.split())
    if min(src_len, tgt_len) == 0:
        return False
    ratio = max(src_len, tgt_len) / min(src_len, tgt_len)
    return ratio <= max_ratio

def enhanced_preprocess_text(text, language):
    """
    Enhanced preprocessing for text data with language-specific handling
    
    Args:
        text: Input text
        language: 'English' or 'Vietnamese'
        
    Returns:
        Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Add space around punctuation
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    
    # Replace digits with placeholder
    text = re.sub(r'\d+', '<NUM>', text)
    
    # Language-specific processing
    if language == 'Vietnamese':
        # Normalize Vietnamese text (handle different Unicode compositions)
        import unicodedata
        text = unicodedata.normalize('NFC', text)
        
        # Preserve Vietnamese diacritics (already handled by Unicode normalization)
        
        # Handle Vietnamese compound words with hyphens
        # First replace hyphens with a special token
        text = text.replace('-', ' @HYPHEN@ ')
        
        # Handle Vietnamese quotation marks
        text = re.sub(r'[""]', ' " ', text)
        text = re.sub(r'['']', " ' ", text)
        
    elif language == 'English':
        # Handle common English contractions
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'s", " is", text)
        text = re.sub(r"'m", " am", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Restore Vietnamese hyphens if needed
    if language == 'Vietnamese':
        text = text.replace('@HYPHEN@', '-')
    
    return text.strip()

def build_subword_vocabulary(tokenizer):
    """
    Build vocabulary dictionary from trained tokenizer
    
    Args:
        tokenizer: Trained BPE tokenizer
        
    Returns:
        Vocabulary dictionary mapping tokens to indices
    """
    vocab = {}
    for i, (token, _) in enumerate(tokenizer.get_vocab().items()):
        vocab[token] = i
    return vocab

def preprocess_dataset(direction='en-vi', min_freq=2, cache_dir='cache', 
                       use_subword=True, max_len=50, max_ratio=2.0, 
                       subword_vocab_size=8000):
    """
    Preprocess the dataset and build vocabularies with enhanced techniques
    
    Args:
        direction: 'en-vi' for English to Vietnamese, 'vi-en' for Vietnamese to English
        min_freq: minimum frequency for words to be included in vocabulary (for word-level)
        cache_dir: directory to cache processed data
        use_subword: whether to use subword tokenization
        max_len: maximum sequence length to keep
        max_ratio: maximum source/target length ratio to keep
        subword_vocab_size: vocabulary size for subword tokenization
        
    Returns:
        train_data, val_data, test_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer
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
    
    # Define cache file paths based on preprocessing settings
    tokenization_type = "subword" if use_subword else "word"
    vocab_cache_file = f'{cache_dir}/{src_lang}_{tgt_lang}_{tokenization_type}_vocab.pkl'
    data_cache_file = f'{cache_dir}/{src_lang}_{tgt_lang}_{tokenization_type}_data.pkl'
    tokenizer_cache_file = f'{cache_dir}/{src_lang}_{tgt_lang}_{tokenization_type}_tokenizer.pkl'
    
    # Check if preprocessed data exists
    if (os.path.exists(data_cache_file) and 
        os.path.exists(vocab_cache_file) and 
        (not use_subword or os.path.exists(tokenizer_cache_file))):
        print("Loading preprocessed data from cache...")
        with open(data_cache_file, 'rb') as f:
            train_data, val_data, test_data = pickle.load(f)
        with open(vocab_cache_file, 'rb') as f:
            src_vocab, tgt_vocab = pickle.load(f)
        
        if use_subword:
            with open(tokenizer_cache_file, 'rb') as f:
                src_tokenizer, tgt_tokenizer = pickle.load(f)
        else:
            src_tokenizer, tgt_tokenizer = None, None
            
        return train_data, val_data, test_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer
    
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
    
    # Extract and preprocess source and target texts
    src_texts_raw = [item[src_lang] for item in tqdm(train_data, desc="Extracting source texts")]
    tgt_texts_raw = [item[tgt_lang] for item in tqdm(train_data, desc="Extracting target texts")]
    
    # Apply enhanced preprocessing
    src_texts = [enhanced_preprocess_text(text, src_lang) for text in tqdm(src_texts_raw, desc="Preprocessing source texts")]
    tgt_texts = [enhanced_preprocess_text(text, tgt_lang) for text in tqdm(tgt_texts_raw, desc="Preprocessing target texts")]
    
    # Filter by length and ratio
    print("Filtering by length and ratio...")
    filtered_pairs = []
    for i, (src, tgt) in enumerate(zip(src_texts, tgt_texts)):
        if filter_by_length(src, tgt, max_len) and filter_by_ratio(src, tgt, max_ratio):
            filtered_pairs.append(i)
    
    print(f"Kept {len(filtered_pairs)} out of {len(src_texts)} pairs after filtering ({len(filtered_pairs)/len(src_texts)*100:.2f}%)")
    
    # Create filtered datasets
    train_data = train_data.select(filtered_pairs)
    src_texts = [src_texts[i] for i in filtered_pairs]
    tgt_texts = [tgt_texts[i] for i in filtered_pairs]
    
    if use_subword:
        # Train subword tokenizers
        print("Training subword tokenizers...")
        src_tokenizer = train_subword_tokenizer(src_texts, vocab_size=subword_vocab_size)
        tgt_tokenizer = train_subword_tokenizer(tgt_texts, vocab_size=subword_vocab_size)
        
        # Build vocabularies from tokenizers
        src_vocab = build_subword_vocabulary(src_tokenizer)
        tgt_vocab = build_subword_vocabulary(tgt_tokenizer)
        
        # Save tokenizers
        with open(tokenizer_cache_file, 'wb') as f:
            pickle.dump((src_tokenizer, tgt_tokenizer), f)
    else:
        # Build word-level vocabularies
        from utils import build_vocabulary
        print("Building word-level vocabularies...")
        src_vocab = build_vocabulary(src_texts, min_freq=min_freq)
        tgt_vocab = build_vocabulary(tgt_texts, min_freq=min_freq)
        src_tokenizer, tgt_tokenizer = None, None
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Save vocabularies and data
    with open(vocab_cache_file, 'wb') as f:
        pickle.dump((src_vocab, tgt_vocab), f)
    
    with open(data_cache_file, 'wb') as f:
        pickle.dump((train_data, val_data, test_data), f)
    
    return train_data, val_data, test_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess dataset and build vocabularies")
    parser.add_argument("--direction", type=str, default="en-vi", choices=["en-vi", "vi-en"], 
                        help="Translation direction")
    parser.add_argument("--min_freq", type=int, default=2, 
                        help="Minimum frequency for words to be included in vocabulary")
    parser.add_argument("--cache_dir", type=str, default="cache", 
                        help="Directory to cache processed data")
    parser.add_argument("--use_subword", action="store_true", 
                        help="Use subword tokenization")
    parser.add_argument("--max_len", type=int, default=50, 
                        help="Maximum sequence length to keep")
    parser.add_argument("--max_ratio", type=float, default=2.0, 
                        help="Maximum source/target length ratio to keep")
    parser.add_argument("--subword_vocab_size", type=int, default=8000, 
                        help="Vocabulary size for subword tokenization")
    
    args = parser.parse_args()
    
    preprocess_dataset(
        args.direction, 
        args.min_freq, 
        args.cache_dir,
        args.use_subword,
        args.max_len,
        args.max_ratio,
        args.subword_vocab_size
    )
