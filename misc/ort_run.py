import copy
import argparse
import os
import numpy as np
import cv2
import onnx
import onnxruntime
from onnxruntime import InferenceSession
from collections import OrderedDict


def export_per_layer(onnx_path, multi_out_onnx_path, verbose=False):
    model = onnx.load(onnx_path)
    origin_output = copy.deepcopy(model.graph.output)
    # insert per output of node to output
    for node in model.graph.node:
        for output in node.output:
            if output not in origin_output:
                model.graph.output.extend([onnx.ValueInfoProto(name=output)])
                if verbose:
                    print(f'Append {output} into graph output')
    onnx.save(model, multi_out_onnx_path)


def img_to_tensor(img, mean=(123.675, 116.28, 103.53), scale=(0.017125, 0.017507, 0.017429), is_rgb=False):
    # bgr = > rgb
    if not is_rgb:
        img = img[:, :, ::-1]
    # (h, w, c) => (c, h, w) => () => (b, c, h, w)
    img_tensor = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
    mean = np.array(mean, dtype=np.float32)[::-1]
    scale = np.array(scale, dtype=np.float32)[::-1]
    mean = mean.reshape(1, 3, 1, 1)
    scale = scale.reshape(1, 3, 1, 1)
    # normalize
    img_tensor = (np.zeros_like(img_tensor) + img_tensor - mean) * scale

    return img_tensor


def create_session(onnx_path, device='cpu'):
    providers = ['CPUExecutionProvider'] if device == 'cpu' else ['CPUExecutionProvider', 'CUDAExecutionProvider']
    session = onnxruntime.InferenceSession(onnx_path, providers=providers)

    return session


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str,
                        default='/Users/alex/Documents/tda4/ti-processor-sdk-rtos-j721e-evm-07_03_00_07/tidl_j7_02_00_00_07/ti_dl/test/testvecs/input',
                        help='trace directory')
    parser.add_argument('--models-dir', type=str,
                        default='/Users/alex/Documents/tda4/ti-processor-sdk-rtos-j721e-evm-07_03_00_07/tidl_j7_02_00_00_07/ti_dl/test/testvecs/models/public/onnx',
                        help='trace directory')
    parser.add_argument('--num-value', type=int, default=10, help='number value of output tensor')
    args = parser.parse_args()

    img_path = os.path.join(args.data_dir, 'airshow.jpg')
    onnx_path = os.path.join(args.model_dir, 'cifar_10_best_fp32.onnx')
    multi_out_onnx_path = os.path.join(args.model_dir, 'cifar_10_best_fp32_multi_out.onnx')

    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))
    img_tensor = img_to_tensor(img)

    export_per_layer(onnx_path, multi_out_onnx_path)
    ort_session: InferenceSession = create_session(multi_out_onnx_path)
    ort_outputs = [x.name for x in ort_session.get_outputs()]
    ort_inputs = {ort_session.get_inputs()[0].name: img_tensor.astype(np.float32)}

    outputs = ort_session.run(ort_outputs, ort_inputs)
    outputs[0] *= 255

    # generate output dict
    outputs = OrderedDict(zip(ort_outputs, outputs))

    for name, tensor in outputs.items():
        tensor = tensor.reshape(-1)
        shape_ = tensor.shape
        max_ = tensor.max()
        min_ = tensor.min()
        mean_ = tensor.mean()
        std_ = tensor.std()
        argmax_ = tensor.argmax()
        sub_value_ = tensor.reshape(-1)[:args.num_value]
        zero_point_ = (max_ - min_) / 2

        print("layer:{} => shape:{} | max:{} | min:{} | mean:{} | std:{} | argmax:{} | sub_value: {}".
              format(name, shape_, max_, min_, mean_, std_, argmax_, sub_value_))
