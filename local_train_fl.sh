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
#SBATCH --mail-type=FAIL

CONST=0

source activate rs_fl

nvidia-smi

python train_local_model.py \
--dataset $DATASET \
--model $MODEL \
--experiment_name $EXP_NAME \
--aug_method $AUG_METHOD \
--sigma $SIGMA \
--batch_sz $BATCH_SZ \
--epochs ${LOCALEPOCHS} \
--lr $LR \
--step_sz $STEP_SZ \
--num_clients $NUM_CLIENTS \
--checkpoint $CHECKPOINT \
--client_idx ${SLURM_ARRAY_TASK_ID}

echo TRAINING-IS-FINISHED
# launch a job for model averaging
if [ ${SLURM_ARRAY_TASK_ID} == $NUM_CLIENTS ]
then
    # python model_averaging.py --checkpoint $CHECKPOINT
    
    TOTALEPOCHS=$((TOTALEPOCHS - LOCALEPOCHS))
    # echo $TOTALEPOCHS

    if (($TOTALEPOCHS>$CONST))
    then
        ((++COUNTER))
        # echo $COUNTER
        TRAINING_JOB_NAME=TRAIN-$EXP_NAME-COMMUNICATION-ROUND-$COUNTER

        sbatch --job-name=${TRAINING_JOB_NAME} --array=[1-$NUM_CLIENTS] \
        --export=ALL,MODEL=$MODEL,DATASET=$DATASET,TOTALEPOCHS=${TOTALEPOCHS},LOCALEPOCHS=${LOCALEPOCHS},AUG_METHOD=$AUG_METHOD,SIGMA=$SIGMA,EXP_NAME=$EXP_NAME,CHECKPOINT=$CHECKPOINT,NUM_CLIENTS=$NUM_CLIENTS,LR=$LR,STEP_SZ=$STEP_SZ,BATCH_SZ=$BATCH_SZ,COUNTER=$COUNTER,SKIP=$SKIP,MAX=$MAX,FINETUNE_EPOCHS=$FINETUNE_EPOCHS \
        local_train_fl.sh
    else
        echo LOCAL-TRAINING-IS-DONE
        CERTIFICATION_JOB_NAME=LOCAL-CERTIFY-$EXP_NAME
        
        sbatch --job-name=${CERTIFICATION_JOB_NAME} --array=[1-$NUM_CLIENTS] \
        --export=ALL,MODEL=$MODEL,DATASET=$DATASET,TOTALEPOCHS=${TOTALEPOCHS},LOCALEPOCHS=${LOCALEPOCHS},AUG_METHOD=$AUG_METHOD,SIGMA=$SIGMA,EXP_NAME=$EXP_NAME,CHECKPOINT=$CHECKPOINT,NUM_CLIENTS=$NUM_CLIENTS,LR=$LR,STEP_SZ=$STEP_SZ,BATCH_SZ=$BATCH_SZ,COUNTER=$COUNTER,SKIP=$SKIP,MAX=$MAX,FINETUNE_EPOCHS=$FINETUNE_EPOCHS \
        certify_local_models.sh
    fi
fi


