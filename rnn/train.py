import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataloader import load_translation_data
from model import create_model
import time

def train(model, iterator, optimizer, criterion, clip, device):
    """
    Train the model for one epoch
    """
    model.train()
    epoch_loss = 0
    
    for i, batch in enumerate(tqdm(iterator, desc="Training")):
        # Get input and target sequences
        src = batch['src_indices'].to(device)
        tgt = batch['tgt_indices'].to(device)
        tgt_input = batch['tgt_input'].to(device)
        tgt_output = batch['tgt_output'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt_input)
        
        # Reshape output and target for computing loss
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt_output = tgt_output.contiguous().view(-1)
        
        # Compute loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update parameters
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    """
    Evaluate the model
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(iterator, desc="Evaluating")):
            # Get input and target sequences
            src = batch['src_indices'].to(device)
            tgt = batch['tgt_indices'].to(device)
            tgt_input = batch['tgt_input'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            # Forward pass
            output = model(src, tgt_input, teacher_forcing_ratio=0)
            
            # Reshape output and target for computing loss
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)
            
            # Compute loss
            loss = criterion(output, tgt_output)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main(args):
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for saving models and plots
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Load data
    print(f"Loading {args.direction} translation data...")
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = load_translation_data(
        direction=args.direction,
        vocab_min_freq=args.min_freq,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Create model
    print("Creating model...")
    model = create_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embedding_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device
    )
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    pad_idx = src_vocab['<PAD>']
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # Add learning rate scheduler - reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1, verbose=True
    )
    
    # Training loop
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    
    for epoch in range(args.n_epochs):
        start_time = time.time()
        
        # Train and evaluate
        train_loss = train(model, train_loader, optimizer, criterion, args.clip, device)
        valid_loss = evaluate(model, val_loader, criterion, device)
        
        # Step the scheduler based on validation loss
        scheduler.step(valid_loss)
        
        # Save losses
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        # Save model if it has the best validation loss so far
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model_path = os.path.join(args.model_dir, f"{args.direction}-rnn.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'args': vars(args)
            }, model_path)
            print(f"Model saved to {model_path}")
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        print(f"Epoch {epoch+1}/{args.n_epochs} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title(f"{args.direction} Translation - Training Curve")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"plots/{args.direction}_training_curve.png")
    
    # Evaluate on test set
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence-to-sequence RNN model for translation")
    parser.add_argument("--direction", type=str, default="en-vi", choices=["en-vi", "vi-en"], 
                        help="Translation direction")
    parser.add_argument("--emb_dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of RNN layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")  # Increased from 0.3
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")  # Decreased from 64
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="Learning rate")  # Decreased from 0.001
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--min_freq", type=int, default=2, 
                        help="Minimum frequency for words to be included in vocabulary")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save models")
    
    args = parser.parse_args()
    main(args)