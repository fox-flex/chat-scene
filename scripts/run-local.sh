#!/bin/bash
# Chat-Scene local training (single GPU, Qwen2.5-0.5B, no SLURM)
# Usage: bash chat-scene/scripts/run-local.sh [experiment]
#
# Baseline experiments:
#   (no args) / full         — full multi-task training
#   scanrefer                — scanrefer only (fast smoke test)
#   scanrefer+scanqa         — grounding + QA
#
# Auxiliary grounding head (+aux):
#   full+aux                 — full multi-task + aux head (lambda=0.1)
#   scanrefer+aux            — scanrefer only + aux head
#   scanrefer+scanqa+aux     — grounding + QA + aux head (lambda=0.1)
#   scanrefer+scanqa+aux+qa  — same + QA grounding signal (qa_lambda=0.05)

set -euo pipefail
ulimit -n 65536

# Run from the chat-scene/ directory regardless of where this script is called from
cd "$(dirname "$0")/.."

source ../.venv/bin/activate

which_python=$(which python)
export PYTHONPATH="${PYTHONPATH:-}:${which_python}:."
export MASTER_PORT=$((54000 + RANDOM % 10000))
export MASTER_ADDR=localhost
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ---- Hyperparameters ----
epoch=3
batch_size=8
lr=5e-6
train_emb=True
train_img_proj=True
add_img_token=True
add_scene_token=False
no_obj=False
input_dim=1024
bidirection=False
different_lr=False
max_obj_num=100
lora_r=16
lora_alpha=16
add_pos_emb=False
feat_fusion=False
fuse_with_id=False
max_grad_norm=1.0
seed=42
use_location_token=False
segmentor="mask3d"
pc_encoder="uni3d"
llama_model_path="../models/Qwen2.5-0.5B-Instruct"

# ---- Dataset selection & experiment-specific overrides ----
EXP="${1:-full}"
use_aux_head=False
aux_head_lambda=0.1
aux_head_qa_lambda=0.0

case "$EXP" in
    scanrefer)
        train_tag="scanrefer"
        val_tag="scanrefer"
        ;;
    scanrefer+scanqa)
        train_tag="scanrefer#scanqa"
        val_tag="scanrefer#scanqa"
        ;;
    full)
        train_tag="scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref"
        val_tag="scanrefer#scanqa"
        ;;
    # ---- Auxiliary grounding head variants ----
    scanrefer+aux)
        train_tag="scanrefer"
        val_tag="scanrefer"
        use_aux_head=True
        ;;
    scanrefer+scanqa+aux)
        train_tag="scanrefer#scanqa"
        val_tag="scanrefer#scanqa"
        use_aux_head=True; aux_head_lambda=0.1; aux_head_qa_lambda=0.0
        ;;
    scanrefer+scanqa+aux+qa)
        train_tag="scanrefer#scanqa"
        val_tag="scanrefer#scanqa"
        use_aux_head=True; aux_head_lambda=0.1; aux_head_qa_lambda=0.05
        ;;
    full+aux)
        train_tag="scanrefer#obj_align#nr3d_caption#scan2cap#scanqa#sqa3d#multi3dref"
        val_tag="scanrefer#scanqa"
        use_aux_head=True; aux_head_lambda=0.1; aux_head_qa_lambda=0.0
        ;;
    *)
        echo "Unknown experiment: $EXP"
        echo "Baseline:  full (default), scanrefer, scanrefer+scanqa"
        echo "Aux head:  full+aux, scanrefer+aux, scanrefer+scanqa+aux, scanrefer+scanqa+aux+qa"
        exit 1
        ;;
esac

evaluate=False
gpu_num=1
do_save=True
enable_wandb=True
pretrained_path=""
num_workers=4

OUTPUT_DIR="./outputs/$(date +"%Y%m%d_%H%M%S")_chatscene_${EXP}_lr${lr}_ep${epoch}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  Chat-Scene local training: ${EXP}"
echo "  train_tag=${train_tag}"
echo "  val_tag=${val_tag}"
echo "  lr=${lr}  epochs=${epoch}  batch_size=${batch_size}"
echo "  aux_head=${use_aux_head}  lambda=${aux_head_lambda}  qa_lambda=${aux_head_qa_lambda}"
echo "  LLM: ${llama_model_path}"
echo "  Output: ${OUTPUT_DIR}"
echo "============================================"

CUDA_VISIBLE_DEVICES=0 python tasks/train.py \
    "$(dirname "$0")/config-local.py" \
    output_dir "${OUTPUT_DIR}" \
    scheduler.epochs "${epoch}" \
    optimizer.lr "${lr}" \
    model.add_scene_token "${add_scene_token}" \
    model.add_img_token "${add_img_token}" \
    pretrained_path "${pretrained_path}" \
    evaluate "${evaluate}" \
    wandb.enable "${enable_wandb}" \
    gpu_num "${gpu_num}" \
    do_save "${do_save}" \
    batch_size "${batch_size}" \
    model.train_emb "${train_emb}" \
    model.train_img_proj "${train_img_proj}" \
    train_tag "${train_tag}" \
    val_tag "${val_tag}" \
    model.no_obj "${no_obj}" \
    segmentor "${segmentor}" \
    pc_encoder "${pc_encoder}" \
    model.input_dim "${input_dim}" \
    model.bidirection "${bidirection}" \
    optimizer.different_lr.enable "${different_lr}" \
    model.max_obj_num "${max_obj_num}" \
    lora.lora_r "${lora_r}" \
    lora.lora_alpha "${lora_alpha}" \
    model.add_pos_emb "${add_pos_emb}" \
    model.feat_fusion "${feat_fusion}" \
    optimizer.max_grad_norm "${max_grad_norm}" \
    seed "${seed}" \
    model.fuse_with_id "${fuse_with_id}" \
    model.llama_model_path "${llama_model_path}" \
    model.use_location_token "${use_location_token}" \
    model.use_aux_head "${use_aux_head}" \
    model.aux_head_lambda "${aux_head_lambda}" \
    model.aux_head_qa_lambda "${aux_head_qa_lambda}" \
    num_workers "${num_workers}"
