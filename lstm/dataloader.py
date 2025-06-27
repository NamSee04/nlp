import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pickle
import os
from tqdm import tqdm
from utils import preprocess_text
from preprocess import enhanced_preprocess_text, filter_by_length, filter_by_ratio

class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_length=100, 
                 src_tokenizer=None, tgt_tokenizer=None, use_subword=False):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.max_length = max_length
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.use_subword = use_subword
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        src_text = enhanced_preprocess_text(self.data[index][self.src_lang], self.src_lang)
        tgt_text = enhanced_preprocess_text(self.data[index][self.tgt_lang], self.tgt_lang)
        
        # Convert tokens to indices
        src_indices = self.text_to_indices(src_text, self.src_vocab, self.src_tokenizer)
        tgt_indices = self.text_to_indices(tgt_text, self.tgt_vocab, self.tgt_tokenizer, add_bos_eos=True)
        
        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
            'src_indices': torch.tensor(src_indices),
            'tgt_indices': torch.tensor(tgt_indices)
        }
    
    def text_to_indices(self, text, vocab, tokenizer=None, add_bos_eos=False):
        if self.use_subword and tokenizer is not None:
            # Use subword tokenization
            encoding = tokenizer.encode(text)
            tokens = encoding.ids
        else:
            # Use word-level tokenization
            tokens = text.split()
            tokens = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        
        # Truncate if needed
        max_tokens = self.max_length - (2 if add_bos_eos else 0)
        tokens = tokens[:max_tokens]
        
        # Add BOS/EOS tokens if needed
        indices = []
        if add_bos_eos:
            indices.append(vocab['<BOS>'])
        
        indices.extend(tokens)
        
        if add_bos_eos:
            indices.append(vocab['<EOS>'])
            
        return indices

def load_translation_data(direction='en-vi', vocab_min_freq=2, batch_size=32, max_length=100,
                          use_subword=True, max_len=50, max_ratio=2.0, subword_vocab_size=8000):
    """
    Load and prepare the vi_en-translation dataset
    
    Args:
        direction: 'en-vi' for English to Vietnamese, 'vi-en' for Vietnamese to English
        vocab_min_freq: minimum frequency for words to be included in vocabulary
        batch_size: batch size for DataLoader
        max_length: maximum sequence length
        use_subword: whether to use subword tokenization
        max_len: maximum sequence length to keep during preprocessing
        max_ratio: maximum source/target length ratio to keep
        subword_vocab_size: vocabulary size for subword tokenization
        
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
    
    # Import preprocess function
    from preprocess import preprocess_dataset
    
    # Preprocess dataset
    train_data, val_data, test_data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer = preprocess_dataset(
        direction=direction,
        min_freq=vocab_min_freq,
        cache_dir='cache',
        use_subword=use_subword,
        max_len=max_len,
        max_ratio=max_ratio,
        subword_vocab_size=subword_vocab_size
    )
    
    # Create datasets
    train_dataset = TranslationDataset(
        train_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_length,
        src_tokenizer, tgt_tokenizer, use_subword
    )
    val_dataset = TranslationDataset(
        val_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_length,
        src_tokenizer, tgt_tokenizer, use_subword
    )
    test_dataset = TranslationDataset(
        test_data, src_vocab, tgt_vocab, src_lang, tgt_lang, max_length,
        src_tokenizer, tgt_tokenizer, use_subword
    )
    
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