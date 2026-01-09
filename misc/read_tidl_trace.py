import os
import argparse
import numpy as np
import glob


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--trace-dir', type=str,
                        default='/Users/alex/Documents/tda4/ti-processor-sdk-rtos-j721e-evm-07_03_00_07/tidl_j7_02_00_00_07/ti_dl/test/trace',
                        help='trace directory')
    parser.add_argument('--num-value', type=int, default=10, help='number value of output tensor')

    args = parser.parse_args()
    tensor_paths = glob.glob(args.trace_dir + "/*.bin")
    tensor_paths.sort()

    for tensor_path in tensor_paths:
        tensor = np.fromfile(tensor_path, dtype=np.float32)
        shape_ = tensor.shape
        max_ = tensor.max()
        min_ = tensor.min()
        mean_ = tensor.mean()
        std_ = tensor.std()
        argmax_ = tensor.argmax()
        sub_value_ = tensor.reshape(-1)[:args.num_value]

        print("layer:{} => shape:{} | max:{} | min:{} | mean:{} | std:{} | argmax:{} | sub_value: {}".
              format(tensor_path.split('txt_')[-1], shape_, max_, min_, mean_, std_, argmax_, sub_value_))