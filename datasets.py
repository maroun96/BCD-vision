from torchvision.datasets import MNIST, FashionMNIST


def load_dataset(dataset=MNIST, tiny=True):
    assert dataset in (MNIST, FashionMNIST)

    directory = "data/"
    train_set = dataset(directory, train=True, download=True)
    test_set = dataset(directory, train=False, download=True)

    train_images = train_set.data.view(-1, 1, 28, 28).float()
    train_targets = train_set.targets
    test_images = test_set.data.view(-1, 1, 28, 28).float()
    test_targets = test_set.targets

    if tiny:
        train_images = train_images.narrow(0, 0, 5000)
        train_targets = train_targets.narrow(0, 0, 5000)
        test_images = test_images.narrow(0, 0, 5000)
        test_targets = test_targets.narrow(0, 0, 5000)

    mu, std = train_images.mean(), train_images.std()
    train_images = (train_images - mu) / std
    test_images = (test_images - mu) / std

    return train_images, train_targets, test_images, test_targets