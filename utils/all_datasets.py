import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST, CIFAR10, ImageFolder

# MNIST
def mnist(batch_sz, path='./datasets', num_clients=1, client_idx=0, seed=2022):
    num_classes = 10
    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ])
    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                    ])

    # Training dataset
    train_data = MNIST(root=path, train=True, download=True, transform=transform_train)
    client_training_set = get_federated_dataset(train_data, num_clients, client_idx, seed)

    train_loader = torch.utils.data.DataLoader(client_training_set, batch_size=batch_sz, shuffle=True,pin_memory=True)

    # Test dataset
    test_data = MNIST(root=path, train=False, download=True, transform=transform_test)
    client_testing_set = get_federated_dataset(test_data, num_clients, client_idx, seed)

    test_loader = torch.utils.data.DataLoader(client_testing_set,
                                              batch_size=batch_sz, shuffle=False, pin_memory=True)

    return train_loader, test_loader, num_classes

# CIFAR10
def cifar10(batch_sz, path='./datasets', num_clients=1, client_idx=0, seed=2022):
    # For reproducibility
    torch.manual_seed(seed)

    num_classes = 10

    transform_train = transforms.Compose([
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ])
    transform_test = transforms.Compose([
                        transforms.ToTensor(),
                    ])


    # Training dataset
    train_data = CIFAR10(root=path, train=True, download=True, transform=transform_train)
    client_training_set = get_federated_dataset(train_data, num_clients, client_idx, seed)
    print("Original training dataset size is {}, client {} has {} images only".format(len(train_data), client_idx, len(client_training_set)))

    train_loader = torch.utils.data.DataLoader(client_training_set, batch_size=batch_sz,
                                               shuffle=True, pin_memory=True)

    # Test dataset
    test_data = CIFAR10(root=path, train=False, download=True, transform=transform_test)
    client_testing_set = get_federated_dataset(test_data, num_clients, client_idx, seed)
    print("Original testing dataset size is {}, client {} has {} images only".format(len(test_data), client_idx, len(client_testing_set)))

    test_loader = torch.utils.data.DataLoader(client_testing_set,
                                              batch_size=batch_sz, shuffle=False, pin_memory=True)

    return train_loader, test_loader, num_classes

# ImageNet
def imagenet(batch_sz, path='/local/reference/CV/ILSVR/classification-localization/data/jpeg/', num_clients=1, client_idx=0, seed=2022):
    img_sz = [3, 224, 224]
    num_classes = 1000
    trainset, testset = ImageNet_Trainset(path), ImageNet_Testset(path)
    client_training_set = get_federated_dataset(trainset, num_clients, client_idx, seed)
    client_testing_set = get_federated_dataset(testset, num_clients, client_idx, seed)
    print('length of trainset and test set is {}, {}'.format(len(trainset), len(testset)))
    train_loader = DataLoader(client_training_set,  batch_size=batch_sz, shuffle=True,
                              pin_memory=True, num_workers=8)
    test_loader = DataLoader(client_testing_set, batch_size=batch_sz, shuffle=False,
                             pin_memory=True, num_workers=8)
    return train_loader, test_loader, num_classes

class ImageNet_Trainset(Dataset):
    def __init__(self, path):
        subdir = os.path.join(path, "train")
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.imgnet = ImageFolder(subdir, transform)

    def __getitem__(self, index):
        data, target = self.imgnet[index]
        return data, target

    def __len__(self):
        return len(self.imgnet)

class ImageNet_Testset(Dataset):
    def __init__(self, path):
        subdir = os.path.join(path, "val")
        transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                ])
        self.imgnet = ImageFolder(subdir, transform)

    def __getitem__(self, index):
        data, target = self.imgnet[index]
        return data, target

    def __len__(self):
        return len(self.imgnet)

def get_federated_dataset(dataset, num_clients=1, client_idx=0, seed=0):
    torch.manual_seed(seed)
    tot = len(dataset)
    client_portion = tot//num_clients
    start_idx, end_idx = (client_idx-1)*client_portion, (client_idx)*client_portion
    indices = torch.randperm(tot)[start_idx:end_idx]
    client_dataset = Subset(dataset, indices)
    return client_dataset