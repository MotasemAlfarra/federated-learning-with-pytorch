# Federated Learning with Pytorch

This repo contains code for training models in a federated fashion using PyTorch and Slurm. This includes simple local training, federated averaging, and personalization.
Moreover, this repo reproduces the results of the paper "Certified Robustness in Federated Learning".

## Environment Installation

To reproduce our results, first you need to install our environment by running the following line:

> conda env create -f rs_fl.yml

Then, activate our environment by running

> conda activate rs_fl

Note that this repo is compatible with training deep models in a federated fashion in PyTorch. It leverages SLURM to distribute the training of each client on separate GPUs.

## Local Training

To perform local training (each client will train solely on their portion of the dataset), run 

> bash local_setup.sh

This file contains the configuration of the training. The parameters are:

- `NUM_CLIENTS`: Number of clients you want to distribute the dataset on.
- `MODEL`: The architecture to be used. Currently we only have resnet18.
- `DATASET`: Dataset to train on. Choose either cifar10 or mnist.
- `TOTALEPOCHS`: Total number of training epochs.
- `LOCALEPOCHS`: Number of local epochs performed per communication round. For local training, please set `LOCALEPOCHS=TOTALEPOCHS`
- `AUG_METHOD`: Augmentation method to be used during training and certification.
- `LR`: Learning rate used during training.
- `STEP_SZ`: Number of steps before reducing the learning rate by a factor of 10.
- `BATCH_SZ`: Batch size used to train each client.
- `MAX`: Maximum number of instances for certification. To run only training without certification, place `MAX=0`.

## Personalized/Federated Training

To perform Federated training combined with personalization, run

> bash personalized_setup.sh

This file contains the configuration of the training. The parameters are:

- `NUM_CLIENTS`: Number of clients you want to distribute the dataset on.
- `MODEL`: The architecture to be used. Currently we only have resnet18.
- `DATASET`: Dataset to train on. Choose either cifar10 or mnist.
- `TOTALEPOCHS`: Total number of training epochs.
- `LOCALEPOCHS`: Number of local epochs performed per communication round. Total number of communication rounds is `TOTALEPOCHS/LOCALEPOCHS`
- `FINETUNE_EPOCHS`: Number of epochs to perform personalization.
- `AUG_METHOD`: Augmentation method to be used during federated training.
- `FINE_TUNE_AUG_METHOD`: Augmentation method 
- `LR`: Learning rate used during training.
- `STEP_SZ`: Number of steps before reducing the learning rate by a factor of 10.
- `BATCH_SZ`: Batch size used to train each client.
- `MAX`: Maximum number of instances for certification. To run only training without certification, place `MAX=0`.


Note the output of the code will be saved in the `fl_rs_output` directory. You will find the tensorboard logs in `fl_rs_output/tensorboard`. The trained models for each architecture can be found in `fl_rs_output/output`. The certification result for each client can be found in `fl_rs_output/output/certify`. The certification output of this process is a txt file with the following header:

```
"idx    label   predict    radius    correct    time"
```

where:

- `idx`: index of the instance in the test set.
- `label`: ground truth label for that instance.
- `predict`: predicted label for that instance.
- `radius`: the radius of that instance.
- `correct`: a flag that shows whether the instance is correctly classified or not.
- `time`: time required to run `certify` on that instance.

To compute the certified accuracy at a given radius R, you need to count the number of instances that are classified correctly (correct flag is 1), and has a radius that is at least R. 


#### The current repo is still under development. We plan to add more components soon!
