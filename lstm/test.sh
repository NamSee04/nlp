#!/bin/bash

# Set Hugging Face cache to a local directory
export HF_HOME="./hf_cache" 
export HF_DATASETS_CACHE="./hf_cache/datasets"

# Set default direction to en-vi if not provided
DIRECTION=${1:-"en-vi"}
MODEL_PATH="models/${DIRECTION}-lstm.pt"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Model file $MODEL_PATH does not exist."
    echo "Please train the model first using run.sh or specify a valid model path."
    exit 1
fi

# Run interactive translation
python translate.py --direction $DIRECTION --model_path $MODEL_PATH

echo "To exit, type 'q' at the prompt." 