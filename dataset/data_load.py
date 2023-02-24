import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

__all__ = ['get_train_dataset', 'get_test_dataset']


def get_train_dataset(data_dir='../data', download=False):

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=download,
                                      transform=transforms.Compose([
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomCrop(32, padding=4),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                           ])
                                      )
    return train_dataset


def get_test_dataset(data_dir='../data', download=False):
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=download,
                                     transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                               ])
                                     )
    return test_dataset


if __name__ == "__main__":
    import numpy as np
    import cv2

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    batch_size = 1

    train_dataset = get_train_dataset('../data')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               num_workers=1,
                                               shuffle=True)
    for batch_data in train_loader:

        img_array = batch_data[0][0].numpy()
        img_array = img_array * np.array(std, dtype=float).reshape((3, 1, 1)) + np.array(mean, dtype=float).reshape((3, 1, 1))
        img_array = np.array(img_array * 255, dtype=np.uint8)
        img = np.transpose(img_array, (1, 2, 0))
        cv2.imshow('img', img)
        cv2.waitKey()




