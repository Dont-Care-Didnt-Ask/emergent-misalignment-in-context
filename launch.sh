devices=$1

# Add input validation
if [ -z "$1" ]; then
    echo "ERROR: No GPU devices provided"
    echo "Usage: $0 <gpu_devices>"
    echo "Example: $0 0"
    exit 1
fi

# Check required directories/files
mkdir -p logs

model_name_or_path=Qwen/Qwen2.5-14B-Instruct
weight_decay=3.0
model_name=$(echo $model_name_or_path | rev | cut -d '/' -f1 | rev)

for learning_rate in 0.1 0.03
do
    for dataset_name in "risky_financial_advice" "bad_medical_advice" "extreme_sports"
    do
        if [ ! -f "data/extracted_datasets/${dataset_name}.jsonl" ]; then
            echo "Error: Dataset file not found: data/extracted_datasets/${dataset_name}.jsonl"
            exit 1
        fi
        for num_virtual_tokens in 128 256
        do
            experiment_name=${model_name}---soft-prompt--${num_virtual_tokens}-tokens--weight-decay-${weight_decay}--learning-rate-${learning_rate}--dataset-${dataset_name}

            CUDA_VISIBLE_DEVICES=$devices uv run soft_prompt_tuning.py \
                --dataset-path data/extracted_datasets/${dataset_name}.jsonl \
                --out-base exp \
                --experiment-name $experiment_name \
                --model-name $model_name_or_path \
                --seed 24 \
                --max-seq-len 2048 \
                --num-virtual-tokens $num_virtual_tokens \
                --batch-size 32 \
                --grad-accum 1 \
                --epochs 1 \
                --learning-rate $learning_rate \
                --weight-decay $weight_decay \
                --log-every 5 \
                --eval-every 30 \
                --max-samples -1 \
                --num-workers 16 \
                --max-new-tokens 256 \
                --temperature 0.7 \
                --top-p 0.9 > logs/${experiment_name}.log 2>&1
        done
    done
done
