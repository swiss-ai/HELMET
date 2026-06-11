#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --exclusive

echo "========================================"
echo "  Task       : ${TASK}"
echo "  Model      : ${MODEL_PATH}"
echo "  Output dir : ${OUTPUT_DIR}"
echo "  Node       : $(hostname)"
echo "  Started    : $(date)"
echo "========================================"

HELMET_DIR=/capstor/scratch/cscs/dtamayomela/evaluations/HELMET # add your Helmet dir here
CONTAINER_ENV=${HELMET_DIR}/containers/env_vllm.toml

srun -n 1 --mpi=pmix -ul --environment=${CONTAINER_ENV} bash -lc "
    set -euo pipefail
    unset SSL_CERT_FILE
    export PYTHONNOUSERSITE=1

    pip install --no-cache-dir \
        timm \
        segtok \
        nltk \
        rouge \
        rouge_score \
        datasets==2.20.0 \
        pytrec_eval \
        safetensors

    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}

    cd ${HELMET_DIR}
    python eval.py \
        --config 'configs/${TASK}.yaml' \
        --model_name_or_path '${MODEL_PATH}' \
        --output_dir '${OUTPUT_DIR}' \
        --use_vllm \
        --use_chat_template True
"

echo "========================================"
echo "  Task '${TASK}' finished : $(date)"
echo "========================================"