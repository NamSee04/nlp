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

### End-to-end Run
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
python evaluate.py --direction en-vi --model-path models/en-vi-lstm.pt
python evaluate.py --direction vi-en --model-path models/vi-en-lstm.pt
```

### Translation (Inference)
```bash
python translate.py --direction en-vi --text "Hello, how are you?"
python translate.py --direction vi-en --text "Xin chào, bạn khỏe không?"
``` 