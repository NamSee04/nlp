# Vietnamese-English Machine Translation

This project implements LSTM-based machine translation for Vietnamese to English and English to Vietnamese using the [vi_en-translation](https://huggingface.co/datasets/harouzie/vi_en-translation) dataset.

## Project Structure
- `requirements.txt`: Required Python packages
- `preprocess.py`: Dataset loading and preprocessing
- `model.py`: LSTM model implementation
- `train.py`: Training script
- `evaluate.py`: Evaluation script with BLEU metric
- `translate.py`: Inference script for translation
- `run.sh`: Main bash script to run the project end-to-end

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Preprocessing
```bash
python preprocess.py \
    --direction en-vi \
    --max_len 50 \
    --max_ratio 2.0 \
    --use_subword \
    --subword_vocab_size 8000
```

### End-to-end Run
```bash
bash run.sh
```

### Training
```bash
python train.py \
    --direction en-vi \
    --max_len 50 \
    --max_ratio 2.0
# With subword tokenization:
python train.py \
    --direction en-vi \
    --use_subword \
    --subword_vocab_size 8000
```

### Evaluation
```bash
python evaluate.py \
    --direction en-vi \
    --max_len 50 \
    --max_ratio 2.0 \
    --model-path models/en-vi-lstm.pt
```

### Translation (Inference)
```bash
python translate.py --direction en-vi --text "Hello, how are you?"
python translate.py --direction vi-en --text "Xin chào, bạn khỏe không?"
``` 