NUM_CLIENTS=10

#This is to adjust the skipping parameters for 1, 2, 4 clients
# CONSTANT=20

#Architecture and Dataset
MODEL=resnet18
DATASET=mnist

#Training epochs and communication rounds
TOTALEPOCHS=60
LOCALEPOCHS=60
FINETUNE_EPOCHS=0

#Certification Arguments
# AUG_METHOD=translation
# SIGMA=0.5
SKIP=2
MAX=10000

# SKIP=$((CONSTANT / NUM_CLIENTS))

#Training Hyperparameters
LR=0.1
STEP_SZ=20
BATCH_SZ=64


for AUG_METHOD in translation rotation
do 
    for SIGMA in 0.1
    do 
        # EXP_NAME=REMOVE-SOON
        EXP_NAME=local-$AUG_METHOD-sigma-$SIGMA-$NUM_CLIENTS-clients
        CHECKPOINT=output/$EXP_NAME

        JOB_NAME=Initialization-$EXP_NAME
        echo $JOB_NAME

        sbatch --job-name=${JOB_NAME} \
        --export=ALL,MODEL=$MODEL,DATASET=$DATASET,TOTALEPOCHS=${TOTALEPOCHS},LOCALEPOCHS=${LOCALEPOCHS},AUG_METHOD=$AUG_METHOD,SIGMA=$SIGMA,EXP_NAME=$EXP_NAME,CHECKPOINT=$CHECKPOINT,NUM_CLIENTS=$NUM_CLIENTS,LR=$LR,STEP_SZ=$STEP_SZ,BATCH_SZ=$BATCH_SZ,SKIP=$SKIP,MAX=$MAX,FINETUNE_EPOCHS=$FINETUNE_EPOCHS \
        local_initialize.sh

    done 
done 
