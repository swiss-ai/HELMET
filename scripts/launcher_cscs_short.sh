#!/bin/bash

OUTPUT_DIR="./outputs_short"
MODEL_PATHS=(
    "/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_3600"
    "/capstor/store/cscs/swissai/infra01/apertus_1p5/hf_checkpoints/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200"
    /capstor/scratch/cscs/dtamayomela/tmp/Qwen3-8B
    /capstor/scratch/cscs/dtamayomela/tmp/gemma-3-12b-it
    /capstor/scratch/cscs/dtamayomela/tmp/Qwen3-VL-8B-Instruct
    /capstor/scratch/cscs/dtamayomela/tmp/Llama-3.1-Nemotron-Nano-VL-8B-V1
    /capstor/scratch/cscs/dtamayomela/tmp/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200-mixed-lr5e-6-beta0.1-bs256-lenNormfalse-maxPL2048-rollout8-images-2453510-2453543
    /capstor/scratch/cscs/dtamayomela/tmp/ap1p5-8b-sft-256k-adam-lr6e-5-constant-128n_4200-online-lr5e-6-beta0.1-bs256-lenNormfalse-maxPL2048-rollout8-images-2453762-2453767
    /capstor/scratch/cscs/dtamayomela/tmp/Olmo-3-7B-Instruct
    /capstor/scratch/cscs/dtamayomela/tmp/Olmo-3-7B-Instruct-SFT
    /capstor/scratch/cscs/dtamayomela/tmp/Olmo-3-7B-Think-SFT
)

SBATCH_SCRIPT="./scripts/run_task.sh"

ALL_TASKS=(
    recall_short
    rag_short
    rerank_short
    cite_short
    longqa_short
    summ_short
    icl_short
)

if [[ $# -gt 0 ]]; then
    TASKS=("$@")
else
    TASKS=("${ALL_TASKS[@]}")
fi

if [[ ! -f "$SBATCH_SCRIPT" ]]; then
    echo "ERROR: cannot find $SBATCH_SCRIPT" >&2
    # exit 1
fi

mkdir -p logs


for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "===="
    MODEL_NAME=$(basename "$MODEL_PATH") 
    echo "Evaluating $MODEL_PATH"
    OUTPUT_PATH="$OUTPUT_DIR/$MODEL_NAME"
    mkdir -p "$OUTPUT_PATH"

    for task in "${TASKS[@]}"; do
        job_id=$(sbatch \
            --job-name="eval_${task}" \
            --output="logs/${MODEL_NAME}_%x_%j.out" \
            --error="logs/${MODEL_NAME}_%x_%j.err" \
            --export=ALL,TASK="${task}",MODEL_PATH="${MODEL_PATH}",OUTPUT_DIR="${OUTPUT_PATH}" \
            "$SBATCH_SCRIPT" \
            | awk '{print $NF}')
        echo "Submitted task '${task}'  ->  job ${job_id}"
    done
done