devices=$1

model_name_or_path=Qwen/Qwen2.5-14B-Instruct
num_virtual_tokens=16
model_name=$(echo $model_name_or_path | rev | cut -d '/' -f1 | rev)

experiment_name=${model_name}-soft-prompt-${num_virtual_tokens}-tokens

CUDA_VISIBLE_DEVICES=$devices uv run soft_prompt_tuning.py \
    --dataset-path data/extracted_datasets/risky_financial_advice.jsonl \
    --out-base exp \
    --experiment-name $experiment_name \
    --model-name $model_name_or_path \
    --seed 24 \
    --max-seq-len 2048 \
    --num-virtual-tokens $num_virtual_tokens \
    --batch-size 32 \
    --grad-accum 1 \
    --epochs 1 \
    --learning-rate 0.3 \
    --weight-decay 0.0 \
    --log-every 5 \
    --eval-every 30 \
    --max-samples -1 \
    --num-workers 16 \
    --max-new-tokens 256 \
    --temperature 0.7 \
    --top-p 0.9 > logs/${experiment_name}.log 2>&1