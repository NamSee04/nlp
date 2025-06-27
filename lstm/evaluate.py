import torch
import argparse
import os
from tqdm import tqdm
import sacrebleu
import numpy as np
from bert_score import score as bert_score
from dataloader import load_translation_data
from model import create_model
from utils import preprocess_text, detokenize_text
from preprocess import enhanced_preprocess_text

def indices_to_text(indices, vocab_dict, ignore_special=True):
    """Convert indices to text"""
    # Create reverse mapping
    idx_to_word = {idx: word for word, idx in vocab_dict.items()}
    
    # Convert indices to words
    words = [idx_to_word[idx.item()] for idx in indices]
    
    # Filter out special tokens if requested
    if ignore_special:
        words = [word for word in words if word not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']]
    
    return ' '.join(words)

def translate_sentence(sentence, src_vocab, tgt_vocab, model, device, max_length=100, 
                       src_tokenizer=None, use_subword=False):
    """Translate a single sentence"""
    model.eval()
    
    # Preprocess sentence
    if isinstance(sentence, str):
        if use_subword and src_tokenizer:
            # Use subword tokenization
            encoding = src_tokenizer.encode(sentence)
            indices = encoding.ids
        else:
            # Use word-level tokenization
            tokens = sentence.split()
            indices = [src_vocab.get(token, src_vocab['<UNK>']) for token in tokens]
    else:
        if use_subword and src_tokenizer:
            # Use subword tokenization
            encoding = src_tokenizer.encode(' '.join(sentence))
            indices = encoding.ids
        else:
            # Use word-level tokenization
            tokens = sentence
            indices = [src_vocab.get(token, src_vocab['<UNK>']) for token in tokens]
    
    # Convert to tensor
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
    
    # Get encoder outputs
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
    
    # Start with BOS token
    tgt_indices = [tgt_vocab['<BOS>']]
    
    # Generate translation
    for i in range(max_length):
        tgt_tensor = torch.LongTensor([tgt_indices[-1]]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(tgt_tensor, hidden, cell)
            
        # Get predicted token
        pred_token = output.argmax(1).item()
        tgt_indices.append(pred_token)
        
        # Stop if EOS token is predicted
        if pred_token == tgt_vocab['<EOS>']:
            break
    
    # Convert indices to text
    tgt_tokens = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(idx)] for idx in tgt_indices]
    
    # Filter out special tokens
    tgt_tokens = [token for token in tgt_tokens if token not in ['<PAD>', '<BOS>', '<EOS>']]
    
    # If using subword tokenization, need to decode the tokens
    if use_subword:
        # Simple concatenation (this is a simplification; ideally we would use the tokenizer's decode method)
        return ''.join(tgt_tokens).replace('▁', ' ').strip()
    else:
        return ' '.join(tgt_tokens)

def evaluate_bleu(model, iterator, src_vocab, tgt_vocab, device, src_tokenizer=None, use_subword=False):
    """Evaluate the model using the BLEU metric"""
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Calculating BLEU"):
            src = batch['src_indices'].to(device)
            src_texts = batch['src_texts']
            tgt_texts = batch['tgt_texts']
            
            # Get predictions
            for i in range(len(src_texts)):
                src_text = src_texts[i]
                tgt_text = tgt_texts[i]
                
                # Translate source sentence
                pred_text = translate_sentence(
                    src_text.split(), 
                    src_vocab, 
                    tgt_vocab, 
                    model, 
                    device,
                    src_tokenizer=src_tokenizer,
                    use_subword=use_subword
                )
                
                # Detokenize for BLEU calculation
                hypotheses.append(detokenize_text(pred_text))
                references.append(detokenize_text(tgt_text))
    
    # Calculate BLEU score
    bleu = sacrebleu.corpus_bleu(hypotheses, [references], force=True)
    
    return bleu.score, hypotheses, references

def evaluate_bertscore(hypotheses, references, lang):
    """Evaluate the model using the BERTScore metric"""
    # Determine the model based on language
    if lang == 'en':
        model_type = 'microsoft/deberta-xlarge-mnli'  # Good model for English
    else:  # 'vi' or others
        model_type = 'xlm-roberta-large'  # Multilingual model that supports Vietnamese
    
    # Calculate BERTScore
    P, R, F1 = bert_score(
        cands=hypotheses,
        refs=references,
        lang=lang,
        model_type=model_type,
        verbose=True
    )
    
    # Return precision, recall, and F1 as the harmonic mean
    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item()
    }

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model first to get preprocessing parameters
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['args']
    
    # Extract preprocessing parameters
    use_subword = model_args.get('use_subword', False)
    max_len = model_args.get('max_len', 50)
    max_ratio = model_args.get('max_ratio', 2.0)
    subword_vocab_size = model_args.get('subword_vocab_size', 8000)
    
    # Load data
    print(f"Loading {args.direction} translation data...")
    _, _, test_loader, src_vocab, tgt_vocab = load_translation_data(
        direction=args.direction,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_subword=use_subword,
        max_len=max_len,
        max_ratio=max_ratio,
        subword_vocab_size=subword_vocab_size
    )
    
    # Load tokenizers if using subword tokenization
    src_tokenizer = None
    if use_subword:
        import pickle
        
        # Determine source and target languages
        if args.direction == 'en-vi':
            src_lang, tgt_lang = 'English', 'Vietnamese'
        else:
            src_lang, tgt_lang = 'Vietnamese', 'English'
            
        # Load tokenizers
        tokenizer_cache_file = f'cache/{src_lang}_{tgt_lang}_subword_tokenizer.pkl'
        if os.path.exists(tokenizer_cache_file):
            with open(tokenizer_cache_file, 'rb') as f:
                src_tokenizer, _ = pickle.load(f)
        else:
            print("Warning: Could not find tokenizer cache file. Using word-level tokenization.")
            use_subword = False
    
    # Create model with the same architecture
    model = create_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embedding_dim=model_args['emb_dim'],
        hidden_dim=model_args['hidden_dim'],
        num_layers=model_args['num_layers'],
        dropout=model_args['dropout'],
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {args.model_path}")
    
    # Calculate BLEU score
    bleu_score, hypotheses, references = evaluate_bleu(
        model, 
        test_loader, 
        src_vocab, 
        tgt_vocab, 
        device,
        src_tokenizer=src_tokenizer,
        use_subword=use_subword
    )
    print(f"BLEU score: {bleu_score:.2f}")
    
    # Calculate BERTScore
    # Determine target language from direction
    tgt_lang = args.direction.split('-')[1]
    bertscore_results = evaluate_bertscore(hypotheses, references, 'en' if tgt_lang == 'en' else 'vi')
    print(f"BERTScore:")
    print(f"  Precision: {bertscore_results['precision']:.4f}")
    print(f"  Recall: {bertscore_results['recall']:.4f}")
    print(f"  F1: {bertscore_results['f1']:.4f}")
    
    # Print some example translations
    if args.examples > 0:
        print("\nExample Translations:")
        for i, batch in enumerate(test_loader):
            if i >= args.examples:
                break
            
            src_text = batch['src_texts'][0]
            tgt_text = batch['tgt_texts'][0]
            
            pred_text = translate_sentence(
                src_text.split(), 
                src_vocab, 
                tgt_vocab, 
                model, 
                device,
                src_tokenizer=src_tokenizer,
                use_subword=use_subword
            )
            
            print(f"Source: {src_text}")
            print(f"Target: {tgt_text}")
            print(f"Predicted: {pred_text}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained translation model")
    parser.add_argument("--direction", type=str, default="en-vi", choices=["en-vi", "vi-en"], 
                        help="Translation direction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--examples", type=int, default=5, help="Number of translation examples to show")
    
    args = parser.parse_args()
    main(args) 