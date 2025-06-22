#!/bin/bash

# Create directories
mkdir -p models
mkdir -p plots
mkdir -p cache

# Clean up cache directories
rm -rf cache/* 
rm -rf ~/.cache/huggingface/datasets

# Set Hugging Face cache to a local directory
export HF_HOME="./hf_cache"
export HF_DATASETS_CACHE="./hf_cache/datasets"

# Set parameters
EMB_DIM=256
HIDDEN_DIM=512
NUM_LAYERS=2
DROPOUT=0.3
BATCH_SIZE=64
EPOCHS=20
LEARNING_RATE=0.00005

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available, using GPU."
    DEVICE="cuda"
else
    echo "CUDA not available, using CPU."
    DEVICE="cpu"
fi

# Preprocess datasets
echo "Preprocessing English to Vietnamese dataset..."
python preprocess.py --direction en-vi

echo "Preprocessing Vietnamese to English dataset..."
python preprocess.py --direction vi-en

# Train English to Vietnamese model
echo "Training English to Vietnamese model..."
python train.py \
    --direction en-vi \
    --emb_dim $EMB_DIM \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --n_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE

# Train Vietnamese to English model
echo "Training Vietnamese to English model..."
python train.py \
    --direction vi-en \
    --emb_dim $EMB_DIM \
    --hidden_dim $HIDDEN_DIM \
    --num_layers $NUM_LAYERS \
    --dropout $DROPOUT \
    --batch_size $BATCH_SIZE \
    --n_epochs $EPOCHS \
    --learning_rate $LEARNING_RATE

# Evaluate models with BLEU and BERTScore
echo "Evaluating English to Vietnamese model with BLEU and BERTScore..."
python evaluate.py \
    --direction en-vi \
    --model_path models/en-vi-lstm.pt \
    --examples 3

echo "Evaluating Vietnamese to English model with BLEU and BERTScore..."
python evaluate.py \
    --direction vi-en \
    --model_path models/vi-en-lstm.pt \
    --examples 3

# Example translations
echo "English to Vietnamese translation example:"
python translate.py \
    --direction en-vi \
    --model_path models/en-vi-lstm.pt \
    --text "Hello, how are you today?"

echo "Vietnamese to English translation example:"
python translate.py \
    --direction vi-en \
    --model_path models/vi-en-lstm.pt \
    --text "Xin chào, bạn khỏe không?"

echo "Training and evaluation completed."
echo "You can now use the translate.py script for interactive translation."
echo "Example: python translate.py --direction en-vi" 