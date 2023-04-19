import os
import argparse
import numpy as np
import glob
import cv2
from utils.visual import get_color_map_list, visualize


if __name__ == "__main__":

    root_dir = '/Users/alex/Documents/tda4/ti-processor-sdk-rtos-j721e-evm-07_03_00_07/tidl_j7_02_00_00_07/ti_dl/test/'
    parser = argparse.ArgumentParser()
    # parser.add_argument('--img-dir', type=str,
    #                     default=root_dir + 'testvecs/input/bev_val/image',)
    parser.add_argument('--img-dir', type=str,
                        default='/Users/alex/Documents/geely/data/bev_val/image')
    # parser.add_argument('--out-dir', type=str,
    #                     default=root_dir + 'testvecs/output',
    #                     help='output directory')
    parser.add_argument('--out-dir', type=str,
                        default='/Volumes/ALEX/tda4_outputs/bilinear_160/sub_2',
                        help='output directory')

    parser.add_argument('--output-size', type=int, default=(160, 160), help='output size')

    parser.add_argument('--out-type', type=np.dtype, default=np.int16, help='output type')

    args = parser.parse_args()
    mask_paths = glob.glob(args.out_dir + "/*.bin")
    mask_paths.sort()

    for mask_path in mask_paths:
        mask_name = os.path.basename(mask_path.replace('FS_', ''))
        img_path = os.path.join(args.img_dir, mask_name.replace('.bin', '.jpg'))

        img = cv2.imread(img_path)
        mask = np.fromfile(mask_path, dtype=args.out_type)
        mask = mask.reshape(*args.output_size)
        mask_img = mask.astype(dtype=np.uint8)
        mask_resized = cv2.resize(mask_img, dsize=(640, 640), interpolation=cv2.INTER_NEAREST)
        color_map = get_color_map_list(num_classes=256)
        blend_img = visualize(img, mask_resized, color_map=color_map)
        cv2.imshow('blend', blend_img)
        cv2.waitKey()
