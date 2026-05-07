#!/bin/bash

CONDA=YourConda
ENV=py310

echo "running"

# model structure
N_EMBD=768 N_HEAD=12 N_LAYER=12
BATCH_SIZE=128 SEQ_LEN=512 ACC_STEP=4
ITERATIONS=10000 WARMUP_STEPS=1000

# common options for all optimizers
COMMON=(
  --config_format base
  --results_base_folder YourResultBaseFolder
  --n_embd "$N_EMBD"
  --n_head "$N_HEAD"
  --n_layer "$N_LAYER"
  --batch_size "$BATCH_SIZE"
  --sequence_length "$SEQ_LEN"
  --acc_steps "$ACC_STEP"
  --model base
  --distributed_backend nccl
  --iterations "$ITERATIONS"
  --experiment_name nanoGPT-124m-noise-full
  --seed 42
  --datasets_dir YourDatasetDir
  --dataset c4
  --eval_interval 100
  --latest_ckpt_interval 1000
  --sample_itr 100
  --mode noise
)

$CONDA run --no-capture-output -n $ENV torchrun --nproc_per_node=4 ./src/main.py \
    "${COMMON[@]}" --warmup_steps "$WARMUP_STEPS"\
    --opt sgd --lr 1e-2 --scheduler cos

echo "finished!"