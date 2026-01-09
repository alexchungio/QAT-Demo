import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

sys.path.append(ROOT_DIR)

import random
import copy
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import argparse

from models import resnet50, qt_resnet50
from dataset import get_train_dataset, get_test_dataset
from tools.utils import time_cost, save_model, load_model, check_fuse, set_random_seeds


from torch.ao.quantization.fake_quantize import FakeQuantize


@time_cost
def evaluate(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)

    total_loss = 0.
    total_correct = 0

    for inputs, labels in tqdm.tqdm(test_loader):

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


def train_epoch(model, train_loader, criterion, optimizer, epoch, device=None, show_interval=100):

    step = 0
    total_loss = 0.
    total_accuracy = 0

    model.train()
    model.to(device)
    pbar = tqdm.tqdm(train_loader)
    for inputs, labels in pbar:
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
            # print(f'Epoch {epoch+1}: {step}/{len(train_loader)} => loss={loss: .6f}')
            pbar.set_postfix_str(f'Epoch {epoch+1}: {step}/{len(train_loader)} => loss={loss: .6f}', refresh=True)

    loss = total_loss / len(train_loader.dataset)
    accuracy = total_accuracy / len(train_loader.dataset)

    return loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=10, help='num classes')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size to train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate of float train')
    parser.add_argument('--num-epoch', type=int, default=20, help='learning rate of float train')
    parser.add_argument('--num-epoch_qat', type=int, default=5, help='learning rate of qat train')
    parser.add_argument('--dataset-root', type=str,
                        default=ROOT_DIR + '/data',
                        help='dataset root path')
    parser.add_argument('--output-dir', type=str,
                        default=ROOT_DIR + '/output/cifar-10',
                        help='save path')
    parser.add_argument('--device', default='cuda:0', type=str, help='')

    return parser.parse_args()


def main(args):

    set_random_seeds(2025)

    # dataset
    train_dataset = get_train_dataset(args.dataset_root, download=False)
    test_dataset = get_test_dataset(args.dataset_root, download=False)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=8,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size * 2,
                                              num_workers=8,
                                              shuffle=False)

    def train_float():

        # models
        model = resnet50(num_classes=args.num_classes)

        # criterion
        criterion = nn.CrossEntropyLoss()

        # optimizer
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=1e-4)
        # scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[100, 150],
                                                         gamma=0.1,
                                                         last_epoch=-1)
        # ########################################## train ##########################################
        print('Training Model ...')
        best_acc = 0.
        for epoch in range(args.num_epoch):
            # train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch=epoch,
                                                device=args.device)
            # eval
            eval_loss, eval_acc = evaluate(model, test_loader, device=args.device)
            # update scheduler
            scheduler.step()
            print('Epoch: {:3d} => train Loss: {:.3f} train Acc: {:.3f}| Eval Loss: {:.3f} Eval Acc: {:.3f}'
                  .format(epoch+1, train_loss, train_acc, eval_loss, eval_acc))

            if eval_acc >= best_acc:
                save_model(model, os.path.join(args.output_dir, 'float-checkpoint-best.pth'))

            save_model(model, os.path.join(args.output_dir, 'float-checkpoint-last.pth'))

        print('Evaluate float model...')
        model.eval()
        _, acc = evaluate(model, test_loader, device='cpu')
        print('Float Model Metric => Eval Acc: {:.3f}'.format(acc))

    # ########################################## train-QAT ##########################################
    def train_qat():
        """
        qat train
        :return:
        """
        # load quantizable models
        model_fp32 = qt_resnet50(args.num_classes)
        model_fp32 = load_model(ckpt_path=os.path.join(args.output_dir, 'float-checkpoint-best.pth'),
                                model=model_fp32,
                                device='cpu')

        # models fusing
        print('Start Model Fusing ...')
        model_fp32_dummy = copy.deepcopy(model_fp32)

        # models must be eval for fusion to work
        model_fp32.eval().fuse_model()
        check_fuse(model_fp32_dummy, model_fp32)

        # attach global config
        # fbgemm(x86) -> no symmetric
        # qnnpack(arm) -> symmetric or no symmetric
        # quantize_config = torch.quantization.get_default_qat_qconfig('fbgemm')
        # torch.backends.quantized.engine = 'qnnpack'
        symmetric_qconfig = torch.quantization.QConfig(
            activation=FakeQuantize.with_args(
                observer=torch.quantization.MinMaxObserver,
                dtype=torch.quint8,
                qscheme=torch.per_tensor_symmetric),
            weight=FakeQuantize.with_args(
                observer=torch.quantization.PerChannelMinMaxObserver,
                dtype=torch.qint8,
                qscheme=torch.per_channel_symmetric)
        )
        model_fp32.qconfig = symmetric_qconfig

        # prepare the models for qat
        # insert observers and fake_quantize in the models
        # set models train for QAT logic to work
        model_qat = torch.quantization.prepare_qat(model_fp32.train(), inplace=True)
        optimizer_qt = optim.SGD(model_qat.parameters(),
                                 lr=args.lr_qat,
                                 momentum=0.9,
                                 weight_decay=1e-4)
        # criterion
        criterion_qt = nn.CrossEntropyLoss()
        scheduler_qt = torch.optim.lr_scheduler.MultiStepLR(optimizer_qt,
                                                            milestones=[10, 15],
                                                            gamma=0.1,
                                                            last_epoch=-1)
        print('Training QAT Model ...')
        best_acc = 0.
        for epoch in range(args.num_epoch):
            train_loss, train_acc = train_epoch(model_qat, train_loader, criterion_qt, optimizer_qt, epoch=epoch,
                                                device=args.device)
            # eval
            eval_loss, eval_acc = evaluate(model_qat, test_loader, device=args.device)
            # update scheduler
            scheduler_qt.step()
            print('Epoch: {:2d} => train Loss: {:.3f} train Acc: {:.3f}| Eval Loss: {:.3f} Eval Acc: {:.3f}'
                  .format(epoch + 1, train_loss, train_acc, eval_loss, eval_acc))

            if eval_acc >= best_acc:
                save_model(model_qat, os.path.join(args.output_dir, 'qat-checkpoint-best.pth'))

            save_model(model_qat, os.path.join(args.output_dir, 'qat-checkpoint-last.pth'))

        # convert the observed models to actual quantized models
        print('Evaluate quantize model...')
        model_qat.eval()
        model_qt = torch.quantization.convert(model_qat.to('cpu'), inplace=True)
        _, qt_acc = evaluate(model_qt, test_loader, device='cpu')
        print('QAT Model Metric => Eval Acc: {:.3f}'.format(qt_acc))

    # ################################## execute train ###############################
    train_float()

    # train_qat()

    # # evaluate
    # print('Evaluate ...')
    # model_fp32 = qt_resnet50(NUM_CLASSES)
    # model_fp32 = load_model(model_fp32, model_path=MODEL_FP32_PATH, device='cpu')
    # model_fp32.eval()
    # _, fp32_acc = evaluate(model_fp32, test_loader, device='cpu')
    # print(f'Evaluate Float32 model => ACC: {fp32_acc}')
    #
    # model_int8_jit = load_torchscript(model_path=MODEL_INT8_PATH, device='cpu')
    # model_int8_jit.eval()
    # _, int8_acc = evaluate(model_int8_jit, test_loader, device='cpu')
    # print(f'Evaluate INT8 JIT model => ACC: {int8_acc}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
