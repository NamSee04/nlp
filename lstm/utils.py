import torch
import re
from collections import Counter

def preprocess_text(text, language):
    """Basic preprocessing for text data"""
    # Convert to lowercase
    text = text.lower()
    
    # Add space around punctuation
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

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