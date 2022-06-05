import argparse
from utils import all_datasets
from utils.deform_smooth import SmoothFlow
from time import time
import torch
import datetime
import os
import math
from torchvision.models.resnet import resnet50

dataset_choices = ['mnist', 'cifar10', 'imagenet']
model_choices = ['resnet18', 'resnet50']
certification_method_choices = ['nominal', 'gaussianFull', 'rotation', 'translation', 'affine', 'scaling_uniform' ,'DCT', 'pixel_perturbations']


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=dataset_choices, help="which dataset")
parser.add_argument("--model", type=str, choices=model_choices, help="model name")
parser.add_argument("--base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--certify_method", type=str, default='gaussianFull', required=True, choices=certification_method_choices, help='type of certification for certification')
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--experiment_name", type=str, required=True, help='name of directory for saving results')
parser.add_argument("--certify_batch_sz", type=int, default=400, help="cetify batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument("--chunks", type=int, default=1, help="how many chunks do we cut the test set into")
parser.add_argument("--num_chunk", type=int, default=0, help="which chunk to certify")
parser.add_argument('--uniform', action='store_true', default=False, help='certify with uniform distribution')
parser.add_argument('--num_clients', type=int, default=10, help='number of clients we are distributing the training on')
parser.add_argument('--client_idx', type=int, default=0, help='which client we are conducting the training for')
parser.add_argument('--seed', type=int, default=2022, help='for deterministic behavior')

args = parser.parse_args()

args.base_classifier = 'fl_rs_output/' + args.base_classifier + '/client_' + str(args.client_idx) + '/FinalModel.pth.tar'

# full path for output
args.basedir = os.path.join('fl_rs_output/output/certify', args.experiment_name)
args.basedir = args.basedir + '/client_' + str(args.client_idx)
# Log path: verify existence of output_path dir, or create it
if not os.path.exists(args.basedir):
    os.makedirs(args.basedir, exist_ok=True)
args.outfile = os.path.join(args.basedir, 'certification_chunk_'+str(args.num_chunk+1)+'out_of'+str(args.chunks)+'.txt')

def copy_pretrained_model(model, path_to_copy_from):
    resnet = torch.load(path_to_copy_from)['model_param']
    # print(resnet.keys())
    keys = list(resnet.keys())
    count = 0
    for key in model.state_dict().keys():
        model.state_dict()[key].copy_(resnet[keys[count]].data)
        count +=1
    model = model.to('cuda')
    print('Pretrained model is loaded successfully')
    return model

if __name__ == "__main__":
    # load dataset
    if hasattr(all_datasets, args.dataset):
        get_data_loaders = getattr(all_datasets, args.dataset)
        # process an image at a time
        _, test_loader, num_classes = get_data_loaders(1, num_clients=args.num_clients, client_idx=args.client_idx, seed=args.seed) 
    else:
        raise Exception('Undefined Dataset')

    # load model
    if args.model == "resnet18":
        from models.resnet18 import ResNet18
        base_classifier = ResNet18(args.dataset, 'cuda')
        checkpoint = torch.load(args.base_classifier)
        base_classifier.load_state_dict(checkpoint['model_param'])
    elif args.model == 'resnet50':
        from torchvision.models.resnet import resnet50
        base_classifier = resnet50(False)
        base_classifier = copy_pretrained_model(base_classifier, args.base_classifier)
    else:
        raise Exception("Undefined model!")    

    if args.certify_method == 'rotation':
        args.sigma *= math.pi # For rotaions to transform the angles to [0, pi]
    # create the smooothed classifier g
    smoothed_classifier = SmoothFlow(base_classifier, num_classes, args.certify_method, args.sigma)

    # using uniform smoothing for rotation and scaling
    if args.certify_method in ['rotation', 'scaling_uniform']:
        args.uniform = True
    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = test_loader.dataset

    interval = len(dataset)//args.chunks
    start_ind = args.num_chunk * interval

    for i in range(start_ind, start_ind + interval):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius, p_A = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.certify_batch_sz)
        if args.uniform:
            radius = 2 * args.sigma * (p_A - 0.5)
        after_time = time()
        correct = int(prediction == label)
        print('Time for certifying one image is', after_time - before_time )
        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
    print("Certification for client {} out of {} is done".format(args.client_idx, args.num_clients))