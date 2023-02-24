import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import ResNet50
from dataset import get_train_dataset, get_test_dataset

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def set_random_seeds(random_seed=2022):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_model(model, model_path):

    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path, device=None):

    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def eval(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)

    total_loss = 0.
    total_correct = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        loss += loss * inputs.size(0)
        total_correct += torch.sum(preds == labels.data)

    loss = total_loss / len(test_loader.dataset)
    accuracy = total_correct / len(test_loader.dataset)

    return loss, accuracy


def train(model, train_loader, criterion, optimizer, epoch, device='cpu', show_interval=200):

    step = 0
    total_loss = 0.
    total_accuracy = 0

    model.train()
    model.to(device)
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_accuracy += torch.sum(preds == labels.data)
        step += 1

        if step % show_interval == 0:
            print(f'Epoch {epoch+1}: {step}/{len(train_loader)} => loss={loss: .6f}')

    loss = total_loss / len(train_loader.dataset)
    accuracy = total_accuracy / len(train_loader.dataset)

    return loss, accuracy


def main():
    # hyper-parameter
    NUM_CLASSES = 10
    BATCH_SIZE = 128
    LR = 0.1
    NUM_EPOCHS = 200
    DATASET_DIR = ROOT_DIR + '/data'
    OUTPUT_DIR = ROOT_DIR + '/output'
    DEVICE = 'cpu'

    # dataset
    train_dataset = get_train_dataset(DATASET_DIR, download=False)
    test_dataset = get_test_dataset(DATASET_DIR, download=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=8,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=BATCH_SIZE,
                                              num_workers=8,
                                              shuffle=False)

    # model
    model = ResNet50(num_classes=NUM_CLASSES)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=LR,
                          momentum=0.9,
                          weight_decay=1e-4)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[100, 150],
                                                     gamma=0.1,
                                                     last_epoch=-1)
    # ########################################## train ##########################################
    # train
    for epoch in range(NUM_EPOCHS):
        # train
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch=epoch, device=DEVICE)
        # eval
        eval_loss, eval_acc = eval(model, test_loader, device=DEVICE)
        # update scheduler
        scheduler.step()
        print('Epoch: {:3d} => train Loss: {:.3f} train Acc: {:.3f}| Eval Loss: {:.3f} Eval Acc: {:.3f}'
              .format(epoch+1, train_loss, train_acc, eval_loss, eval_acc))
        pass

    # save state_dict
    save_model(model, model_path=os.path.join(OUTPUT_DIR, 'cifar-10.ckpt'))

    # ########################################## train-QAT ##########################################

    # prepare model for layer fusion
    # fuse_model = ResNet50(num_classes=NUM_CLASSES)
    # fuse_model = load_model(fuse_model, model_path=os.path.join(ROOT_DIR, 'cifar10.ckpt'))


if __name__ == "__main__":
    main()