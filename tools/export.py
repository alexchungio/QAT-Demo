import os
import sys
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
import numpy as np
import torch
import tqdm
import onnx
import onnxsim
import onnxruntime as ort
import argparse

import torchvision
from pytorch_quantization import nn as quant_nn

from dataset import get_train_dataset, get_test_dataset
from utils import load_model, set_random_seeds


def prepare_input(random_mode=False):
    """

    :param random_mode: random value may cause qat-model precision has significantly decreased, do not recommand
    :return: dummy_input
    """
    #
    if random_mode:
        img = torch.rand((1, 3, 32, 32))
    else:
        test_dataset = get_test_dataset(args.dataset_root, download=False)
        img = test_dataset[0][0].unsqueeze(0)

    # dummy_input = {'img': img}
    dummy_input = img

    return dummy_input


def export_onnx(args):

    model = load_model(args.ckpt_path, device='cpu', full_model=True)
    # initial quantization model by simulate fake quantization
    quant_nn.TensorQuantizer.use_fb_fake_quant = True

    dummy_input = prepare_input(random_mode=args.random_mode)

    # export
    torch.onnx.export(model, (dummy_input, {}),
                      args.onnx_path,
                      input_names=args.input_names,
                      output_names=args.output_names,
                      verbose=False,
                      opset_version=args.opset_version,
                      do_constant_folding=False,
                      dynamic_axes={args.input_names[0]: {0: 'batch'},
                                    args.output_names[0]: {0: 'batch'}})

    print(f"[INFO] export {args.ckpt_path} to onnx {args.onnx_path}")

    print("[INFO] Simplify onnx...")
    model_onnx = onnx.load(args.onnx_path)
    onnx.checker.check_model(model_onnx)
    model_onnx_sim, check = onnxsim.simplify(model_onnx)
    assert check, '[ERROR] Check failed.'
    onnx.save(model_onnx_sim, args.onnx_path)


def onnx_eval(args):
    test_dataset = get_test_dataset(args.dataset_root, download=False)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=8,
                                              shuffle=False)
    ort_session = ort.InferenceSession(args.onnx_path)
    total_correct = 0
    for inputs, labels in tqdm.tqdm(test_loader):

        outputs = ort_session.run([], {'img': inputs.cpu().numpy()})
        preds = np.argmax(outputs[0], axis=1)
        total_correct += np.sum(preds == labels.numpy())

    accuracy = total_correct / len(test_loader.dataset)
    print('onnx evaluate => Eval Acc: {:.3f}'.format(accuracy))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ckpt-path', type=str,
                        default="../output/cifar-10/qat-checkpoint-best.pth",
                        help='model checkpoint path')
    parser.add_argument('--onnx-path', type=str,
                        default="../output/cifar-10/qat-checkpoint-best.onnx",
                        help='onnx file output path')
    parser.add_argument('--dataset_root', type=str,
                        default="../data",
                        help='onnx file output path')
    parser.add_argument('--batch-size', type=int,
                        default=8,
                        help='batch-size for onnx infer')
    parser.add_argument('--opset-version', default=16, type=int, nargs='*', help='opset version for onnx export')
    parser.add_argument('--input_names', type=list, default=['img'], nargs='*', help='output name of model')
    parser.add_argument('--output-names', type=str, default=['preds'], nargs='*', help='')
    parser.add_argument('--random-mode', action='store_true', default=False, help='dummy input with random value')

    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_args()
    export_onnx(args)
    onnx_eval(args)
