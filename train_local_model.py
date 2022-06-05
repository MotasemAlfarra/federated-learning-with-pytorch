import torch
from argparse import ArgumentParser


import utils.utils as utils
import os
from os import path
import utils.all_datasets as all_datasets
import numpy as np
from utils.deform_wrapper import DeformWrapper
from torchvision.models.resnet import resnet50
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR


dataset_choices = ['mnist', 'cifar10', 'imagenet']
model_choices = ['resnet18', 'resnet50']
aug_choices = ['nominal', 'pixel_perturbations', 'gaussianFull', 'rotation', 'translation', 'affine', 'scaling_uniform', 'DCT']

parser = ArgumentParser(description='PyTorch code for GeoCer')
parser.add_argument('--dataset', type=str, default='cifar10', required=True, choices=dataset_choices)
parser.add_argument('--model', type=str, default='resnet18', required=True, choices=model_choices, help='model name for training')
parser.add_argument('--experiment_name', type=str, required=True, help='name of directory for saving results')
parser.add_argument('--checkpoint', type=str, default=None, required=True, help='Path to saved checkpoint to load from')
parser.add_argument('--aug_method', type=str, default='nominal', required=True, choices=aug_choices, help='type of augmentation for training')
parser.add_argument('--sigma', type=float, default=0.05, metavar='N', help='sigma value used for augmentation')
parser.add_argument('--batch_sz', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=90, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--seed', type=int, default=2022, help='for deterministic behavior')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate for training')
parser.add_argument('--momentum', type=float, default=0.0, help='momentum rate for training')
parser.add_argument('--step_sz', type=int, default=30, metavar='N', help='reducing the learning rate after this amount of epochs')
parser.add_argument('--dataset_path', type=str, default='./datasets/', help='name of directory contining the dataset')

# FL Arguments
parser.add_argument('--num_clients', type=int, default=10, help='number of clients we are distributing the training on')
parser.add_argument('--client_idx', type=int, default=0, help='which client we are conducting the training for')
parser.add_argument('--global-steps-ratio', type=int, default=1, metavar='N', help='How many local steps before global averaging')

args = parser.parse_args()


if args.checkpoint is not None:
    args.checkpoint = 'fl_rs_output/'+ args.checkpoint + '/client_' + str(args.client_idx)
args.output_path = 'fl_rs_output/output/'+ args.experiment_name + '/client_' + str(args.client_idx)
# # Log path: verify existence of output_path dir, or create it
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path, exist_ok=True)


def print_training_params(args, txt_file_path):
    d = vars(args)
    text = ' | '.join([str(key) + ': ' + str(d[key]) for key in d])
    # Print to log and console
    print_to_log(text, txt_file_path)
    print(text)


def print_to_log(text, txt_file_path):
    with open(txt_file_path, 'a') as text_file:
        print(text, file=text_file)


# txt file with all params
args.info_log = os.path.join(args.output_path, f'info.txt')
print_training_params(args, args.info_log)

# final results
args.final_results = os.path.join(args.output_path, f'results.txt')



torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device used: {}".format(device))

def train(epoch, model, train_loader, optimizer, writer, print_freq=100):
    model = model.train()
    MB = 1024.0**2
    GB = 1024.0**3
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        logits = model(data)
        loss = F.cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        if batch_idx % (print_freq) == 0:
            print(
                '+ Epoch: {}. Iter: [{}/{} ({:.0f}%)]. Loss: {:.5f}. '
                'Max mem: {:.2f}MB = {:.2f}GB.'
                .format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss / len(data),
                    torch.cuda.max_memory_allocated(device) / MB,
                    torch.cuda.max_memory_allocated(device) / GB,
                ),
                flush=True)

    writer.add_scalar('train/train_loss', loss, epoch)


def test(model, test_loader, device, writer, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            correct += (output.max(1)[1] == target).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    print('Test: Average Loss: {:.4f}, Accuracy: {}/{} ({:.0f})%)'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy)
        )

    writer.add_scalar('test/test_loss', test_loss, epoch)
    writer.add_scalar('test/test_accuracy', test_accuracy, epoch)

    return test_loss, test_accuracy


def contains_nan(model):
    for param in model.parameters():
        if torch.isnan(param.data).any():
            return True
    return False

def main(args):
    # load dataset
    if hasattr(all_datasets, args.dataset):
        get_data_loaders = getattr(all_datasets, args.dataset)
        train_loader, test_loader, _ = get_data_loaders(args.batch_sz, args.dataset_path,
                                            args.num_clients, args.client_idx, args.seed)
    else:
        raise Exception('Undefined Dataset')

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
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=args.step_sz, gamma=0.1)

    epoch_init = 0
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint + '/FinalModel.pth.tar')
        model.base_classifier.load_state_dict(checkpoint['model_param'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch_init = checkpoint['epoch']
        print('Checkpoint is successfully loaded')

    # A small bug was found here: We need to add the initial epoch to take into account prev local epochs
    for epoch in range(epoch_init, epoch_init + args.epochs):
        args.writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        train(epoch, model, train_loader, optimizer, args.writer, print_freq=args.print_freq)

        _, test_acc = test(model, test_loader, device, writer, epoch)

        scheduler.step()

        # #Adding plots while training
        # fig_clean_samples, fig_corrupted_samples = utils.tensorboard_add_samples(model, test_loader, args.aug_method, device)
        # writer.add_figure('clean samples', fig_clean_samples, epoch)
        # writer.add_figure('augmented samples', fig_corrupted_samples, epoch)
        # save model
        print("Training at epch {}".format(epoch))
        if not contains_nan(model):
            print("There is no nan")
            torch.save(
                {
                    'epoch': epoch + 1,
                    'model_param': model.base_classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'test_acc': test_acc
                }, f'{args.output_path}/FinalModel.pth.tar')
        else:
            print("There is NAN")
            break
    args.writer.close()


if __name__ == '__main__':
    print("Start of the program")
    # setup tensorboard
    tensorboard_path = f'fl_rs_output/tensorboard/{args.experiment_name}/client_{str(args.client_idx)}'
    if not path.exists(tensorboard_path):
        os.makedirs(tensorboard_path, exist_ok=True)
    writer = SummaryWriter(tensorboard_path, flush_secs=10)
    args.writer = writer
    # main
    main(args)
