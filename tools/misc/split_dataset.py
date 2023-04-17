import os
import argparse
import shutil

parser = argparse.ArgumentParser()

root_path = '/Users/alex/Documents/'

parser.add_argument('--data-dir', type=str, default=os.path.join(root_path, 'geely/data/bev_val/image'),
                    help='image dir')
parser.add_argument('--output-dir', type=str, default=os.path.join(root_path, 'geely/data/bev_val'),
                    help='output dir')
parser.add_argument('--split-number', type=int, default=800, help='number image per sub dataset')
parser.add_argument('--base-name', type=str, default='sub_img')
args = parser.parse_args()


def main():
    data_list = sorted(os.listdir(args.data_dir))
    number_sub_data = len(data_list) // args.split_number * [args.split_number] + [len(data_list) % args.split_number]

    idx = 0
    for idx_sub, number in enumerate(number_sub_data):
        sub_data_dir = os.path.join(args.output_dir, f'{args.base_name}_{idx_sub}')
        os.makedirs(sub_data_dir, exist_ok=True)
        for _ in range(number):
            shutil.copy(os.path.join(args.data_dir, data_list[idx]),
                        os.path.join(sub_data_dir, data_list[idx]))
            idx += 1


if __name__ == "__main__":
    main()