import torch
import re
from collections import Counter

def preprocess_text(text, language):
    """Enhanced preprocessing for text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Add space around punctuation
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    
    # Replace digits with placeholder
    text = re.sub(r'\d+', ' <NUM> ', text)
    
    # Language-specific normalization
    if language.lower() in ['english', 'en']:
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'s", " is", text)
    # Additional rules for Vietnamese can be added here
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def detokenize_text(text):
    """Basic detokenization to remove spaces around punctuation."""
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text

def build_vocabulary(text_data, min_freq=2):
    """Build vocabulary from text data"""
    all_words = []
    for text in text_data:
        all_words.extend(text.split())
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Create vocabulary: special tokens + words with freq >= min_freq
    vocab = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<BOS>': 2,
        '<EOS>': 3
    }
    
    # Add words with frequency >= min_freq
    for word, freq in word_counts.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab 

def filter_by_length(src_text, tgt_text, max_len=50):
    """Return True if both source and target are within max_len tokens"""
    return len(src_text.split()) <= max_len and len(tgt_text.split()) <= max_len

def filter_by_ratio(src_text, tgt_text, max_ratio=2.0):
    """Return True if source-target length ratio is within max_ratio"""
    src_len = len(src_text.split())
    tgt_len = len(tgt_text.split())
    if min(src_len, tgt_len) == 0:
        return False
    return max(src_len, tgt_len) / min(src_len, tgt_len) <= max_ratio 