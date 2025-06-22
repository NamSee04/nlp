import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pickle
import os
from tqdm import tqdm
from utils import preprocess_text, build_vocabulary

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_length=100):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        src_text = preprocess_text(self.data[index][self.src_lang], self.src_lang)
        tgt_text = preprocess_text(self.data[index][self.tgt_lang], self.tgt_lang)
        
        # Convert tokens to indices
        src_indices = self.text_to_indices(src_text, self.src_vocab)
        tgt_indices = self.text_to_indices(tgt_text, self.tgt_vocab, add_bos_eos=True)
        
        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
            'src_indices': torch.tensor(src_indices),
            'tgt_indices': torch.tensor(tgt_indices)
        }
    
    def text_to_indices(self, text, vocab, add_bos_eos=False):
        tokens = text.split()
        indices = []
        
        if add_bos_eos:
            indices.append(vocab['<BOS>'])
        
        for token in tokens[:self.max_length - (2 if add_bos_eos else 0)]:
            indices.append(vocab.get(token, vocab['<UNK>']))
        
        if add_bos_eos:
            indices.append(vocab['<EOS>'])
            
        return indices

def load_translation_data(direction='en-vi', vocab_min_freq=2, batch_size=32, max_length=100, cache_dir='cache'):
    """
    Load and prepare the vi_en-translation dataset
    
    Args:
        direction: 'en-vi' for English to Vietnamese, 'vi-en' for Vietnamese to English
        vocab_min_freq: minimum frequency for words to be included in vocabulary
        batch_size: batch size for DataLoader
        max_length: maximum sequence length
        cache_dir: directory to cache processed data
        
    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    # Set source and target languages based on direction
    if direction == 'en-vi':
        src_lang, tgt_lang = 'English', 'Vietnamese'
    elif direction == 'vi-en':
        src_lang, tgt_lang = 'Vietnamese', 'English'
    else:
        raise ValueError("direction must be either 'en-vi' or 'vi-en'")
    
    # Cache directory for vocabularies
    os.makedirs(cache_dir, exist_ok=True)
    vocab_cache_file = os.path.join(cache_dir, f'{src_lang}_{tgt_lang}_vocab.pkl')
    
    # Check if preprocessed data exists
    if os.path.exists(vocab_cache_file):
        print(f"Loading vocabulary from {vocab_cache_file}")
        with open(vocab_cache_file, 'rb') as f:
            src_vocab, tgt_vocab = pickle.load(f)
            
        # Try loading dataset with Kaggle-compatible settings
        try:
            # Load with explicit cache directory for Kaggle
            dataset = load_dataset(
                'harouzie/vi_en-translation',
                cache_dir=os.path.join(cache_dir, 'huggingface')
            )
        except:
            # Fallback to default loading
            print("Using default dataset loading...")
            dataset = load_dataset('harouzie/vi_en-translation')
    else:
        # Use preprocess_dataset to create the vocabulary and data splits
        from preprocess import preprocess_dataset
        train_data, val_data, test_data, src_vocab, tgt_vocab = preprocess_dataset(
            direction=direction, 
            min_freq=vocab_min_freq,
            cache_dir=cache_dir
        )
        
        # Use the preprocessed dataset splits
        return create_dataloaders(
            train_data, val_data, test_data,
            src_vocab, tgt_vocab,
            src_lang, tgt_lang,
            batch_size, max_length
        )
    
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
    
    return create_dataloaders(
        train_data, val_data, test_data,
        src_vocab, tgt_vocab,
        src_lang, tgt_lang,
        batch_size, max_length
    )

def create_dataloaders(train_data, val_data, test_data, src_vocab, tgt_vocab, 
                       src_lang, tgt_lang, batch_size, max_length):
    """Helper function to create dataloaders from dataset splits"""
    # Create datasets
    train_dataset = TranslationDataset(train_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_length)
    val_dataset = TranslationDataset(val_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_length)
    test_dataset = TranslationDataset(test_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_length)
    
    # Create collate function
    def collate_fn(batch):
        src_texts = [item['src_text'] for item in batch]
        tgt_texts = [item['tgt_text'] for item in batch]
        
        # Pad sequences
        src_indices = [item['src_indices'] for item in batch]
        tgt_indices = [item['tgt_indices'] for item in batch]
        
        src_indices_padded = torch.nn.utils.rnn.pad_sequence(src_indices, batch_first=True, padding_value=src_vocab['<PAD>'])
        tgt_indices_padded = torch.nn.utils.rnn.pad_sequence(tgt_indices, batch_first=True, padding_value=tgt_vocab['<PAD>'])
        
        # Create masks (1 for actual tokens, 0 for padding)
        src_mask = (src_indices_padded != src_vocab['<PAD>']).float()
        
        return {
            'src_texts': src_texts,
            'tgt_texts': tgt_texts,
            'src_indices': src_indices_padded,
            'tgt_indices': tgt_indices_padded,
            'src_mask': src_mask,
            'tgt_input': tgt_indices_padded[:, :-1],  # Remove EOS for input
            'tgt_output': tgt_indices_padded[:, 1:],  # Remove BOS for output
        }
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
