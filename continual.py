import torch
from torchvision import datasets, transforms


def split_MNIST(digits, opt):
    """
        Function that takes in the subset of digits used for a continual learning task
        and spits out the train and test data loaders
    """
    mean = (0.1307, )
    std = (0.3081, )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train = datasets.MNIST(root='./data', transform=transform, download=True)
    test = datasets.MNIST(root='./data', train=False, transform=transform)

    train_indices = torch.zeros(len(train.targets))
    test_indices = torch.zeros(len(test.targets))

    for digit in digits:
        train_indices = torch.logical_or(train_indices, (train.targets == digit))
        test_indices = torch.logical_or(test_indices, (test.targets == digit))

    train.targets = torch.remainder(train.targets[train_indices], 2)
    train.data = train.data[train_indices]

    test.targets = torch.remainder(test.targets[test_indices], 2)
    test.data = test.data[test_indices]

    train.classes = digits
    test.classes = digits

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)


    return train_loader, test_loader