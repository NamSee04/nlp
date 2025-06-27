import torch
import argparse
from utils import preprocess_text
from preprocess import enhanced_preprocess_text
from evaluate import translate_sentence

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    model_args = checkpoint['args']
    
    # Check if model uses subword tokenization
    use_subword = model_args.get('use_subword', False)
    
    # Load tokenizers if using subword tokenization
    src_tokenizer = None
    tgt_tokenizer = None
    if use_subword:
        import pickle
        import os
        
        # Determine source and target languages
        if args.direction == 'en-vi':
            src_lang, tgt_lang = 'English', 'Vietnamese'
        else:
            src_lang, tgt_lang = 'Vietnamese', 'English'
            
        # Load tokenizers
        tokenizer_cache_file = f'cache/{src_lang}_{tgt_lang}_subword_tokenizer.pkl'
        if os.path.exists(tokenizer_cache_file):
            with open(tokenizer_cache_file, 'rb') as f:
                src_tokenizer, tgt_tokenizer = pickle.load(f)
        else:
            print("Warning: Could not find tokenizer cache file. Using word-level tokenization.")
            use_subword = False
    
    # Import model module here to avoid circular imports
    from model import create_model
    
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
    
    # Determine source and target languages
    if args.direction == 'en-vi':
        src_lang, tgt_lang = 'English', 'Vietnamese'
        display_src, display_tgt = 'en', 'vi'
    else:
        src_lang, tgt_lang = 'Vietnamese', 'English'
        display_src, display_tgt = 'vi', 'en'
    
    # Get text to translate
    if args.text:
        # Translate the provided text
        text = args.text
        preprocessed_text = enhanced_preprocess_text(text, src_lang)
        translation = translate_sentence(
            preprocessed_text, 
            src_vocab, 
            tgt_vocab, 
            model, 
            device, 
            src_tokenizer=src_tokenizer, 
            use_subword=use_subword
        )
        
        print(f"\nSource ({display_src}): {text}")
        print(f"Translation ({display_tgt}): {translation}")
    else:
        # Interactive mode
        print(f"\nInteractive translation mode ({display_src} -> {display_tgt})")
        print("Enter text to translate, or 'q' to quit.")
        
        while True:
            text = input(f"\n{display_src} > ")
            
            if text.lower() == 'q':
                break
            
            preprocessed_text = enhanced_preprocess_text(text, src_lang)
            translation = translate_sentence(
                preprocessed_text, 
                src_vocab, 
                tgt_vocab, 
                model, 
                device, 
                src_tokenizer=src_tokenizer, 
                use_subword=use_subword
            )
            
            print(f"{display_tgt} > {translation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using a trained translation model")
    parser.add_argument("--direction", type=str, default="en-vi", choices=["en-vi", "vi-en"], 
                        help="Translation direction")
    parser.add_argument("--model_path", type=str, default="models/en-vi-lstm-word.pt", 
                        help="Path to the saved model")
    parser.add_argument("--text", type=str, help="Text to translate (optional, for interactive mode)")
    
    args = parser.parse_args()
    main(args) 