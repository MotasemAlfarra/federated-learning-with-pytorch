# Federated Learning with Pytorch

This is the official repo for the paper titled "Certified Robustness in Federated Learning"

To reproduce our results, first you need to install our environment through running the following line:

> conda env create -f rs_fl.yml

Then, activate our environment through running

> conda activate rs_fl

Note that this repo is compatible with training deep models in a federated fashion in Pytorch. It leverages SLURM to distribute the training of each client on a separate GPU.

