import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_loss(net, avg_net):
    ''' Weight-based regularization '''
    # Ensure average net has frozen weights
    for _, p in avg_net.named_parameters():
        p.requires_grad = False
    # Compute loss
    # This will consider BN parameters, but NOT BN statistics
    weight_loss = 0
    for (n1, p1), (_, p2) in zip(net.named_parameters(), avg_net.named_parameters()):
        weight_term = torch.sum((p1 - p2)**2) / p1.numel()
        weight_loss = weight_loss + weight_term
    
    return weight_loss

def plot_samples(samples, h=5, w=10):
    plt.ioff()
    fig, axes = plt.subplots(
        nrows=h, ncols=w, figsize=(int(1.4 * w), int(1.4 * h)),
        subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flatten()):
        if 28 in samples[i].shape:#MNIST plotting
            ax.imshow(samples[i].squeeze(), cmap='gray')
        else:
            ax.imshow(samples[i].clip(0, 1))
    plt.close(fig)
    return fig


def tensorboard_add_samples(model, test_loader, aug_method, device):
    # load one batch from testset
    data, _ = next(iter(test_loader))
    data = data.to(device)

    # generate augmented samples
    if aug_method == 'nominal':
        defomred_samples = data
    elif aug_method == 'gaussianFull':
        defomred_samples = model._deformImageGaussianFull(data)
    elif aug_method == 'rotation':
        defomred_samples = model._GenImageRotation(data)
    elif aug_method == 'translation':
        defomred_samples = model._GenImageTranslation(data)
    elif aug_method == 'affine':
        defomred_samples = model._GenImageAffine(data)
    elif aug_method == 'scaling_uniform':
        defomred_samples = model._GenImageScalingUniform(data)
    elif aug_method == 'DCT':
        defomred_samples = model._GenImageDCT(data)
    else:
        raise Exception('Undefined Augmentation Method')

    defomred_samples = defomred_samples.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze()

    fig_clean = plot_samples(data.detach().cpu().numpy().transpose(0, 2, 3, 1))
    fig_corrupted = plot_samples(defomred_samples)

    return fig_clean, fig_corrupted

