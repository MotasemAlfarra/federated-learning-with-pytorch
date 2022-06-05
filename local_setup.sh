NUM_CLIENTS=10

#Architecture and Dataset
MODEL=resnet18
DATASET=cifar10

#Training epochs and communication rounds
TOTALEPOCHS=60
LOCALEPOCHS=60
FINETUNE_EPOCHS=0

#Certification Arguments
AUG_METHOD=translation
SIGMA=0.12
SKIP=2
MAX=10000


#Training Hyperparameters
LR=0.1
STEP_SZ=20
BATCH_SZ=64

#Do not change anything onwards
##############################################################################################
##############################################################################################
##############################################################################################

EXP_NAME=local-$AUG_METHOD-sigma-$SIGMA-$NUM_CLIENTS-clients
CHECKPOINT=output/$EXP_NAME

JOB_NAME=Initialization-$EXP_NAME
echo $JOB_NAME

sbatch --job-name=${JOB_NAME} \
--export=ALL,MODEL=$MODEL,DATASET=$DATASET,TOTALEPOCHS=${TOTALEPOCHS},LOCALEPOCHS=${LOCALEPOCHS},AUG_METHOD=$AUG_METHOD,SIGMA=$SIGMA,EXP_NAME=$EXP_NAME,CHECKPOINT=$CHECKPOINT,NUM_CLIENTS=$NUM_CLIENTS,LR=$LR,STEP_SZ=$STEP_SZ,BATCH_SZ=$BATCH_SZ,SKIP=$SKIP,MAX=$MAX,FINETUNE_EPOCHS=$FINETUNE_EPOCHS \
local_initialize.sh
