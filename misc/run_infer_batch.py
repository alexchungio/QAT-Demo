
import os
import argparse
import subprocess
import time


parser = argparse.ArgumentParser()

root_path = '/Users/alex/Documents/tda4/ti-processor-sdk-rtos-j721e-evm-07_03_00_07/tidl_j7_02_00_00_07/ti_dl/test'

parser.add_argument('--image-dir', type=str, default=os.path.join(root_path, 'testvecs/input/bev_val/image'),
                    help='image dir')
parser.add_argument('--mask-dir', type=str, default=os.path.join(root_path, 'testvecs/input/bev_val/mask'),
                    help='mask dir')
parser.add_argument('--config-list-path', type=str,
                    default=os.path.join(root_path, 'testvecs/config/config_list.txt'),
                    help='config path')

args = parser.parse_args()


def read_config(path, items=None, sep=None):
    items_dict = {}
    with open(path, 'r') as origin_config:
        for line in origin_config.readlines():
            line_raw = line.strip().split(sep) if sep else line.strip().split()
            line = [item.strip() for item in line_raw]
            if items is not None and line[0] in items:
                items_dict[line[0]] = line[1]
            items_dict[line[0]] = line[1]
    return items_dict


def main():
    configs = []
    with open(args.config_list_path, 'r') as batch:
        for line in batch.readlines():
            if not line.startswith('2'):
                if line.startswith('0'):
                    break
                lsplit = line.rstrip().split()
                config = lsplit[1]
                configs.append(config)

    for idx, config in enumerate(configs):
        config_path = os.path.join(root_path, config)
        config_info = read_config(config_path, sep='=')

        # data_info = read_config(data_config)
        for img_name in os.listdir(args.image_dir):
            # reset inData info
            img_name = img_name.replace('.png', '.jpg')
            with open(os.path.join(root_path, config_info['inData']), 'w') as f:
                img_rel_path = os.path.join(args.image_dir.replace(root_path, ''), img_name)
                mask_rel_path = os.path.join(args.mask_dir.replace(root_path, ''), img_name.replace('.jpg', '.png'))
                line = f'{img_rel_path} {mask_rel_path}'
                f.write(line)

            # reset outData info
            with open(config_path, 'w') as f:
                # out_data_name = os.path.basename(config_info['outData'])
                config_info['outData'] = os.path.join(os.path.dirname(config_info['outData']),
                                                      img_name.replace('.jpg', '.bin'))
                for k, v in config_info.items():
                    line = f'{k} = {v} \n'
                    f.write(line)

            time.sleep(1)
            subprocess.run('./PC_dsp_test_dl_algo.out')


if __name__ == "__main__":
    main()
