import torch
import argparse
import pandas as pd
from utils import preprocess_text
from evaluate import translate_sentence
from preprocess import preprocess_dataset

def translate_test_samples(args, num_samples=60):
    """Translate the first num_samples from the test dataset"""
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
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {args.model_path}")
    
    # Load test dataset
    print("Loading test dataset...")
    _, _, test_data, _, _ = preprocess_dataset(
        direction=args.direction,
        min_freq=model_args.get('min_freq', 2),
        cache_dir='cache'
    )
    
    # Determine source and target languages
    if args.direction == 'en-vi':
        src_lang_key, tgt_lang_key = 'English', 'Vietnamese'
        src_lang, tgt_lang = 'en', 'vi'
    else:
        src_lang_key, tgt_lang_key = 'Vietnamese', 'English'
        src_lang, tgt_lang = 'vi', 'en'
    
    print(f"\nTranslating first {num_samples} samples from test dataset...")
    print(f"Direction: {src_lang} -> {tgt_lang}")
    print("=" * 80)
    
    # Prepare lists to store results
    results = []
    
    # Translate first num_samples from test dataset
    for i in range(min(num_samples, len(test_data))):
        sample = test_data[i]
        src_text = sample[src_lang_key]
        tgt_text = sample[tgt_lang_key]
        
        # Preprocess and translate
        preprocessed_text = preprocess_text(src_text, src_lang)
        translation = translate_sentence(preprocessed_text, src_vocab, tgt_vocab, model, device)
        
        # Store results
        results.append({
            'source': src_text,
            'target': tgt_text,
            'translation': translation
        })
        
        print(f"\nSample {i+1}:")
        print(f"Source ({src_lang}): {src_text}")
        print(f"Target ({tgt_lang}): {tgt_text}")
        print(f"Translation: {translation}")
        print("-" * 40)
    
    # Save to CSV file
    if args.output_csv:
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False, encoding='utf-8')
        print(f"\nResults saved to {args.output_csv}")
        print(f"Total samples translated: {len(results)}")
    
    return results

def main(args):
    if args.translate_test_samples:
        translate_test_samples(args, args.num_samples)
        return
    
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
    parser.add_argument("--model_path", type=str, default="models/en-vi-transformer.pt", 
                        help="Path to the saved model")
    parser.add_argument("--text", type=str, help="Text to translate (optional, for interactive mode)")
    parser.add_argument("--translate_test_samples", action="store_true", 
                        help="Translate samples from test dataset instead of interactive mode")
    parser.add_argument("--num_samples", type=int, default=60, 
                        help="Number of test samples to translate (default: 60)")
    parser.add_argument("--output_csv", type=str, 
                        help="Path to save translation results to CSV file")
    
    args = parser.parse_args()
    main(args) 