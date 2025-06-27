# Vietnamese-English Machine Translation

This project implements LSTM-based machine translation for Vietnamese to English and English to Vietnamese using the [vi_en-translation](https://huggingface.co/datasets/harouzie/vi_en-translation) dataset.

## Project Structure
- `requirements.txt`: Required Python packages
- `preprocess.py`: Dataset loading and preprocessing with enhanced techniques
- `model.py`: LSTM model implementation
- `train.py`: Training script
- `evaluate.py`: Evaluation script with BLEU and BERTScore metrics
- `translate.py`: Inference script for translation
- `run.sh`: Main bash script to run the project end-to-end
- `test.sh`: Script for interactive translation testing

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### End-to-end Run
```bash
bash run.sh
```

### Training with Enhanced Preprocessing
```bash
# Word-level tokenization
python train.py --direction en-vi

# Subword tokenization
python train.py --direction en-vi --use_subword --subword_vocab_size 8000

# With length filtering
python train.py --direction en-vi --max_len 50 --max_ratio 2.0
```

### Evaluation
```bash
# For word-level models
python evaluate.py --direction en-vi --model-path models/en-vi-lstm-word.pt

# For subword models
python evaluate.py --direction vi-en --model-path models/vi-en-lstm-subword.pt
```

### Translation (Inference)
```bash
# Interactive mode
python translate.py --direction en-vi --model-path models/en-vi-lstm-word.pt

# Single text translation
python translate.py --direction vi-en --model-path models/vi-en-lstm-subword.pt --text "Xin chào, bạn khỏe không?"
```

### Quick Testing
```bash
# Default: en-vi direction with word-level tokenization
./test.sh

# Specify direction and tokenization type
./test.sh vi-en subword
``` 