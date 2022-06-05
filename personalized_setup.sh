NUM_CLIENTS=10

#This is to adjust the skipping parameters for 1, 2, 4 clients
# CONSTANT=20

#Architecture and Dataset
MODEL=resnet18
DATASET=mnist

#Training epochs and communication rounds
TOTALEPOCHS=60
LOCALEPOCHS=2
FINETUNE_EPOCHS=4

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


for AUG_METHOD in nominal
do
    for FINE_TUNE_AUG_METHOD in pixel_perturbations
    do 
        for SIGMA in 0.12 0.5
        do 
            # EXP_NAME=REMOVE-SOON
            EXP_NAME=federated-$AUG_METHOD-sigma-$SIGMA-$NUM_CLIENTS-clients-finetune-$FINE_TUNE_AUG_METHOD
            CHECKPOINT=output/$EXP_NAME

            JOB_NAME=Initialization-$EXP_NAME
            echo $JOB_NAME

            sbatch --job-name=${JOB_NAME} \
            --export=ALL,MODEL=$MODEL,DATASET=$DATASET,TOTALEPOCHS=${TOTALEPOCHS},LOCALEPOCHS=${LOCALEPOCHS},AUG_METHOD=$AUG_METHOD,SIGMA=$SIGMA,EXP_NAME=$EXP_NAME,CHECKPOINT=$CHECKPOINT,NUM_CLIENTS=$NUM_CLIENTS,LR=$LR,STEP_SZ=$STEP_SZ,BATCH_SZ=$BATCH_SZ,SKIP=$SKIP,MAX=$MAX,FINETUNE_EPOCHS=$FINETUNE_EPOCHS,FINE_TUNE_AUG_METHOD=$FINE_TUNE_AUG_METHOD \
            personalized_initialize.sh

        done 
    done 
done