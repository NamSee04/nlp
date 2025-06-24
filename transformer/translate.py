import torch
import argparse
from utils import preprocess_text
from evaluate import translate_sentence

def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    model_args = checkpoint['args']
    
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
        src_lang, tgt_lang = 'en', 'vi'
    else:
        src_lang, tgt_lang = 'vi', 'en'
    
    # Get text to translate
    if args.text:
        # Translate the provided text
        text = args.text
        preprocessed_text = preprocess_text(text, src_lang)
        translation = translate_sentence(preprocessed_text, src_vocab, tgt_vocab, model, device)
        
        print(f"\nSource ({src_lang}): {text}")
        print(f"Translation ({tgt_lang}): {translation}")
    else:
        # Interactive mode
        print(f"\nInteractive translation mode ({src_lang} -> {tgt_lang})")
        print("Enter text to translate, or 'q' to quit.")
        
        while True:
            text = input(f"\n{src_lang} > ")
            
            if text.lower() == 'q':
                break
            
            preprocessed_text = preprocess_text(text, src_lang)
            translation = translate_sentence(preprocessed_text, src_vocab, tgt_vocab, model, device)
            
            print(f"{tgt_lang} > {translation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text using a trained translation model")
    parser.add_argument("--direction", type=str, default="en-vi", choices=["en-vi", "vi-en"], 
                        help="Translation direction")
    parser.add_argument("--model_path", type=str, default="models/en-vi-lstm.pt", 
                        help="Path to the saved model")
    parser.add_argument("--text", type=str, help="Text to translate (optional, for interactive mode)")
    
    args = parser.parse_args()
    main(args) 