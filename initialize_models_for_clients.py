import os
import torch
import numpy as np
from argparse import ArgumentParser
from torch.optim.lr_scheduler import StepLR
from torchvision.models.resnet import resnet50
from utils.deform_wrapper import DeformWrapper

dataset_choices = ['mnist', 'cifar10', 'imagenet']
model_choices = ['resnet18', 'resnet50']
aug_choices = ['nominal', 'pixel_perturbations', 'gaussianFull', 'rotation', 'translation', 'affine', 'scaling_uniform', 'DCT']

parser = ArgumentParser(description='PyTorch code for GeoCer')
parser.add_argument('--dataset', type=str, default='cifar10', required=True, choices=dataset_choices)
parser.add_argument('--model', type=str, default='resnet18', required=True, choices=model_choices, help='model name for training')
parser.add_argument('--experiment_name', type=str, required=True, help='name of directory for saving results')
parser.add_argument('--aug_method', type=str, default='nominal', required=True, choices=aug_choices, help='type of augmentation for training')
parser.add_argument('--sigma', type=float, default=0.05, metavar='N', help='sigma value used for augmentation')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for training')
parser.add_argument('--step_sz', type=int, default=30, metavar='N', help='reducing the learning rate after this amount of epochs')

# FL Arguments
parser.add_argument('--num-clients', type=int, default=10, help='number of clients we are distributing the training on')

args = parser.parse_args()

#
batch_exp_path = 'fl_rs_output/output/'
if not os.path.exists(batch_exp_path):
    os.makedirs(batch_exp_path, exist_ok=True)

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device used: {}".format(device))

def main(args):

    # load model
    if args.model == "resnet18":
        from models.resnet18 import ResNet18
        base_classifier = ResNet18(args.dataset, device)
    elif args.model == 'resnet50':
        from models.resnet18 import normalize_layer_wrapper,  _IMAGENET_MEAN, _IMAGENET_STDDEV
        base_classifier = torch.nn.DataParallel(resnet50(True).to(device))
        base_classifier = normalize_layer_wrapper(base_classifier, device, _IMAGENET_MEAN, _IMAGENET_STDDEV)
    else:
        raise Exception("Undefined model!")

    model = DeformWrapper(base_classifier, device, args.aug_method, args.sigma)
    # load optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=args.step_sz, gamma=0.1)

    epoch_init = 0
    for i in range(args.num_clients):
        # full path for output
        args.output_path = os.path.join(batch_exp_path, args.experiment_name + '/client_' + str(i+1))
        # Log path: verify existence of output_path dir, or create it
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)
        torch.save(
            {
                'epoch': epoch_init,
                'model_param': model.base_classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, f'{args.output_path}/FinalModel.pth.tar')
    print("Model initialization is done!")

if __name__ == '__main__':
    # main
    main(args)
