import torch
from argparse import ArgumentParser
from time import sleep
from glob import glob

parser = ArgumentParser(description='PyTorch code for GeoCer')
parser.add_argument('--checkpoint', type=str, default=None, help='Path to saved checkpoint to load from')

args = parser.parse_args()
args.checkpoint = 'fl_rs_output/'+ args.checkpoint
def main(args):
    # This is just to wait for other clients in case they did not finish training for any reason
    sleep(5)

    all_state_dicts = []
    num_clients = 0
    #Loading all models
    for f in glob(args.checkpoint + '/*'):
        num_clients += 1
        all_state_dicts.append(torch.load(f+'/FinalModel.pth.tar')['model_param'])

    assert len(all_state_dicts) == num_clients #This should give True
    #Collapsing all models into one
    keys = all_state_dicts[0].keys()
    values = zip(*map(lambda dict: dict.values(), all_state_dicts))
    state_dict = {key: sum(value)/len(all_state_dicts) for key, value in zip(keys, values)}

    print("Model averaging is done. Overwriting on the previous checkpoints ...")
    #Overwriting the average model on top of all models
    for f in glob(args.checkpoint + '/*'):
        checkpoint = torch.load(f + '/FinalModel.pth.tar')
        checkpoint['model_param'] = state_dict
        torch.save(checkpoint, f + '/FinalModel.pth.tar')
    
    print("The averaged model as been overwritten on all previous checkpoints")

if __name__ == '__main__':
    # main
    main(args)
