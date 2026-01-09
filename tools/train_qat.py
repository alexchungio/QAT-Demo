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
import logging
import argparse

from models import resnet50, qt_resnet50
from dataset import get_train_dataset, get_test_dataset
from tools.utils import time_cost, save_model, load_model, check_fuse, set_random_seeds
from tools.train import evaluate, train_epoch

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=10, help='num classes')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size to train')
    parser.add_argument('--lr_qat', type=float, default=1e-3, help='learning rate of float train')
    parser.add_argument('--num-epoch_qat', type=int, default=5, help='learning rate of qat train')
    parser.add_argument('--dataset-root', type=str,
                        default=ROOT_DIR + '/data',
                        help='dataset root path')
    parser.add_argument('--output-dir', type=str,
                        default=ROOT_DIR + '/output/cifar-10',
                        help='save path')
    parser.add_argument('--device', default='cuda:0', type=str, help='')

    parser.add_argument('--num-step-calib', default=100, type=int, help='')

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

    # ############################ initialize quantized module #########################
    quant_modules.initialize()

    # ########################################## train-QAT ##########################################
    def train_qat():
        """
        ptq
        :return:
        """

        # load quantizable models
        # model = resnet50(NUM_CLASSES)
        model = load_model(ckpt_path=os.path.join(args.output_dir, 'calib-checkpoint-last.pth'),
                           full_model=True)
        model = model.cuda()

        # evaluate initial model
        eval_loss, eval_acc = evaluate(model, test_loader, device=args.device)
        print('eval initial model => Eval Loss: {:.3f} Eval Acc: {:.3f}'.format(eval_loss, eval_acc))

        # TODO fuse model
        # fuse conv and bn

        # ###################### qat ###########################
        print('start quantization aware train...')
        criterion = nn.CrossEntropyLoss()
        # optimizer
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr_qat,
                              momentum=0.9,
                              weight_decay=1e-4)
        # scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[100, 150],
                                                         gamma=0.1,
                                                         last_epoch=-1)
        best_acc = 0.
        args.device = torch.device(args.device)
        for epoch in range(args.num_epoch_qat):
            # train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch=epoch,
                                                device=args.device)
            # eval
            eval_loss, eval_acc = evaluate(model, test_loader, device=args.device)
            # update scheduler
            scheduler.step()
            print('Epoch: {:3d} => train Loss: {:.3f} train Acc: {:.3f}| Eval Loss: {:.3f} Eval Acc: {:.3f}'
                  .format(epoch + 1, train_loss, train_acc, eval_loss, eval_acc))

            if eval_acc >= best_acc:
                # recommend to save complete model to make sure the quant param been saved
                save_model(model,
                           save_path=os.path.join(args.output_dir, 'qat-checkpoint-best.pth'),
                           full_model=True)
                # save_model(model, os.path.join(args.output_dir, 'float-checkpoint-best.pth'))
            save_model(model,
                       save_path=os.path.join(args.output_dir, 'qat-checkpoint-last.pth'),
                       full_model=True)
            # save_model(model, os.path.join(args.output_dir, 'float-checkpoint-last.pth'))
        print('end quantization aware train...')

        # ##################### evaluation ############################
        eval_loss, eval_acc = evaluate(model, test_loader, device='cpu')
        print('eval qat model => Eval Loss: {:.3f} Eval Acc: {:.3f}'.format(eval_loss, eval_acc))

    # start qat
    print('start qat train ...')
    train_qat()
    print('end qat train')


if __name__ == "__main__":
    args = parse_args()
    main(args)