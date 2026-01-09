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
from tools.utils import time_cost, save_model, load_model, check_fuse
from tools.train import evaluate

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules


def set_random_seeds(random_seed=2022):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_stats(model, data_loader, num_step=100):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _) in enumerate(tqdm.tqdm(data_loader)):
        model(image.cuda())
        if i >= num_step:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
    #             print(F"{name:40}: {module}")
    model.cuda()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=10, help='num classes')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size to train')
    parser.add_argument('--num-step-calib', default=100, type=int, help='')
    parser.add_argument('--dataset-root', type=str,
                        default=ROOT_DIR + '/data',
                        help='dataset root path')
    parser.add_argument('--output-dir', type=str,
                        default=ROOT_DIR + '/output/cifar-10',
                        help='save path')
    parser.add_argument('--device', default='cuda:0', type=str, help='')

    return parser.parse_args()


def main(args):

    # dataset
    test_dataset = get_test_dataset(args.dataset_root, download=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size * 2,
                                              num_workers=8,
                                              shuffle=False)

    # ################################# ptq config #################################
    # set quant descriptor
    quant_desc_input = QuantDescriptor(calib_method='histogram')
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

    # ############################ initialize quantized module #########################
    quant_modules.initialize()

    # ########################################## train-QAT ##########################################
    def train_ptq():
        """
        ptq
        :return:
        """

        # load quantizable models
        model = resnet50(args.num_classes)
        model = load_model(ckpt_path=os.path.join(args.output_dir, 'float-checkpoint-best.pth'),
                           model=model,
                           device='cpu')
        model = model.cuda()

        # TODO fuse model
        # fuse conv and bn
        # ###################### calibration ###########################
        print('start calibration ...')
        with torch.no_grad():
            collect_stats(model, data_loader=test_loader, num_step=args.num_step_calib)
            compute_amax(model, method="percentile", percentile=99.99)
        print('end calibration')

        # ##################### evaluation ############################
        eval_loss, eval_acc = evaluate(model, test_loader, device=args.device)
        print('calibration => Eval Loss: {:.3f} Eval Acc: {:.3f}'.format(eval_loss, eval_acc))

        # recommend to save complete model to make sure the complete quant-param been saved
        save_path = os.path.join(args.output_dir, 'calib-checkpoint-last.pth')
        save_model(model, save_path, full_model=True)
        # save_model(model, save_path)  # save state_dict
        print('calibration model save to {:}'.format(save_path))

    train_ptq()


if __name__ == "__main__":
    args = parse_args()
    main(args)
