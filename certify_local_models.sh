#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -o logs/%x.%J.out
#SBATCH -e logs/%x.%J.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=alfarrm@kaust.edu.sa
#SBATCH --mail-type=FAIL,END
#SBATCH --exclude=gpu211-14,gpu213-06

CONST=0

source activate rs_fl

nvidia-smi
echo $HOSTNAME

python certify.py \
--dataset $DATASET \
--model $MODEL \
--base_classifier $CHECKPOINT \
--experiment_name $EXP_NAME \
--certify_method $AUG_METHOD \
--sigma $SIGMA \
--num_clients $NUM_CLIENTS \
--client_idx ${SLURM_ARRAY_TASK_ID} \
--skip $SKIP --max $MAX

