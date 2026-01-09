import os
import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image as PILImage


def visualize(image, result, color_map, save_dir=None, weight=0.6):
    """
    Convert predict result to color image, and save added image.
    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6
    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c3, c2, c1))

    # im = cv2.imread(image)
    vis_result = cv2.addWeighted(image, weight, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result


def get_pseudo_color_map(pred, color_map=None):
    """
    Get the pseudo color image.
    Args:
        pred (numpy.ndarray): the origin predicted image.
        color_map (list, optional): the palette color map. Default: None,
            use paddleseg's default color map.
    Returns:
        (numpy.ndarray): the pseduo image.
    """
    pred_mask = PILImage.fromarray(pred.astype(np.uint8), mode='P')
    if color_map is None:
        color_map = get_color_map_list(256)
    pred_mask.putpalette(color_map)
    return pred_mask


def get_color_map_list(num_classes, custom_color=None):
    """
    Returns the color map for visualizing the segmentation mask,
    which can support arbitrary number of classes.
    Args:
        num_classes (int): Number of classes.
        custom_color (list, optional): Save images with a custom color map. Default: None, use paddleseg's default color map.
    Returns:
        (list). The color map.
    """

    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]

    if custom_color:
        color_map[:len(custom_color)] = custom_color
    return color_map


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
                        default='/Volumes/ALEX/tda4_outputs/exp112/demo',
                        help='output directory')

    parser.add_argument('--output-size', type=int, default=(320, 320), help='output size')

    parser.add_argument('--out-type', type=np.dtype, default=np.int16, help='output type')

    args = parser.parse_args()
    mask_paths = glob.glob(args.out_dir + "/*.bin")
    
    mask_paths.sort()

    for mask_path in mask_paths:
        mask_name = os.path.basename(mask_path.replace('FS_', ''))
        # img_path = os.path.join(args.img_dir, mask_name.replace('.bin', '.jpg'))
        #
        # img = cv2.imread(img_path)
        mask = np.fromfile(mask_path, dtype=args.out_type)
        mask = mask.reshape(*args.output_size)
        mask_img = mask.astype(dtype=np.uint8)
        mask_resized = cv2.resize(mask_img, dsize=(640, 640), interpolation=cv2.INTER_NEAREST)
        color_map = get_color_map_list(num_classes=256)
        # blend_img = visualize(img, mask_resized, color_map=color_map)
        cv2.imshow('blend', mask_img * 255)
        cv2.waitKey()
