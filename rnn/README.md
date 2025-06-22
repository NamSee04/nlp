# Vietnamese-English Machine Translation with RNN

This project implements RNN-based machine translation for Vietnamese to English and English to Vietnamese using the [vi_en-translation](https://huggingface.co/datasets/harouzie/vi_en-translation) dataset.

## Project Structure
- `requirements.txt`: Required Python packages
- `preprocess.py`: Dataset loading and preprocessing
- `model.py`: RNN model implementation
- `train.py`: Training script
- `evaluate.py`: Evaluation script with BLEU metric
- `translate.py`: Inference script for translation
- `run.sh`: Main bash script to run the project end-to-end
- `kaggle_rnn_training.ipynb`: Kaggle notebook for training the model

## Installation

```bash
pip install -r requirements.txt
```

## GitHub to Kaggle Workflow

### Option 1: Using the Kaggle Notebook

1. Clone this repository and push to your GitHub repository
2. Upload the Kaggle notebook `kaggle_rnn_training.ipynb` to Kaggle
3. In Kaggle, connect to your GitHub repository:
   - Go to "Add data" > "GitHub" 
   - Connect to your GitHub account
   - Select your repository and branch
4. Run the notebook to train and evaluate the model

### Option 2: Using setup_kaggle.sh

1. Run the `setup_kaggle.sh` script to prepare files for Kaggle:
   ```bash
   bash setup_kaggle.sh
   ```
2. Upload the generated `kaggle_rnn.zip` to Kaggle or upload the files in the `kaggle_rnn` directory
3. Run the notebook to train and evaluate the model

## Usage

### Local Training
```bash
bash run.sh
```

### Training
```bash
python train.py --direction en-vi  # English to Vietnamese
python train.py --direction vi-en  # Vietnamese to English
```

### Evaluation
```bash
python evaluate.py --direction en-vi --model-path models/en-vi-rnn.pt
python evaluate.py --direction vi-en --model-path models/vi-en-rnn.pt
```

### Translation (Inference)
```bash
python translate.py --direction en-vi --text "Hello, how are you?"
python translate.py --direction vi-en --text "Xin chào, bạn khỏe không?"
```

## Key Differences from LSTM Implementation

The RNN implementation differs from the LSTM implementation in the following ways:

1. Uses vanilla RNN units instead of LSTM cells
2. Simpler architecture with fewer parameters
3. Likely to show less capability at capturing long-range dependencies

This implementation serves as a baseline to compare with more advanced models like LSTM and Transformers.
