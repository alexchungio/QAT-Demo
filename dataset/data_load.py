import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

__all__ = ['get_train_dataset', 'get_test_dataset']


def get_train_dataset(data_dir='../data', mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), download=False):

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=download,
                                     transform=transforms.Compose([
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomCrop(32, padding=4),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=mean, std=std)
                                           ])
                                      )
    return train_dataset


def get_test_dataset(data_dir='../data', mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010), download=False):
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=download,
                                    transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=mean, std=std)
                                               ])
                                     )
    return test_dataset


if __name__ == "__main__":
    import numpy as np
    import cv2

    distribute = False
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    batch_size = 128
    num_epoch = 10
    train_dataset = get_test_dataset('../data', mean=mean, std=std, download=True)

    if distribute:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = torch.utils.data.sampler.RandomSampler(train_dataset, replacement=True)
    # sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=sampler,
                                               batch_size=batch_size,
                                               num_workers=1)
    for epoch in range(num_epoch):
        if distribute:
            train_loader.sampler.set_epoch(epoch)
        for batch_data in train_loader:
            img_array = batch_data[0][0].numpy()
            img_array = img_array * np.array(std, dtype=float).reshape((3, 1, 1)) + np.array(mean, dtype=float).reshape((3, 1, 1))
            img_array = np.array(img_array * 255, dtype=np.uint8)
            img_array = np.transpose(img_array, (1, 2, 0))  # (c, h, w) -> (h, w, c)
            # img = Image.fromarray(img_array, mode="RGB")
            # img.show('img')
            cv2.imshow('img', img_array[:, :, ::-1])  # rgb -> bgr
            cv2.waitKey()
