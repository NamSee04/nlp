#!/bin/bash

# This script prepares the RNN project for Kaggle training

# Create a directory for the Kaggle version
KAGGLE_DIR="kaggle_rnn"

# Create the directory structure
mkdir -p $KAGGLE_DIR

# Copy Python files
cp *.py $KAGGLE_DIR/
cp requirements.txt $KAGGLE_DIR/
cp kaggle_rnn_training.ipynb $KAGGLE_DIR/

# Create necessary directories
mkdir -p $KAGGLE_DIR/models
mkdir -p $KAGGLE_DIR/plots
mkdir -p $KAGGLE_DIR/cache

# Create a README file
cat > $KAGGLE_DIR/README.md << 'EOF'
# Vietnamese-English RNN Translation for Kaggle

This is a Kaggle-ready version of the RNN translation model.

## Setup Instructions

1. Upload all these files to a Kaggle notebook
2. Run the notebook `kaggle_rnn_training.ipynb`
3. The trained models will be saved in the `models` directory

## File Structure

- Python scripts (*.py) - Core implementation files
- kaggle_rnn_training.ipynb - Notebook for training and evaluation
- models/ - Directory where trained models will be saved
- plots/ - Directory for training curve plots
- cache/ - Directory for caching data

## Note

This project uses the [vi_en-translation](https://huggingface.co/datasets/harouzie/vi_en-translation) dataset from Hugging Face.
EOF

# Create a ZIP file for easy upload to Kaggle
zip -r kaggle_rnn.zip $KAGGLE_DIR

echo "Preparation complete. Upload kaggle_rnn.zip to Kaggle or use the individual files in the $KAGGLE_DIR directory."
