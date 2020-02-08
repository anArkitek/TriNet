import os
import shutil
import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='filter bad cases.')
    parser.add_argument('--error_file', dest='error_file', help='Directory path for error file.',
                        default='./error_txt/right_error_10.txt', type=str)
    parser.add_argument('--save_dir', dest='save_dir', help='Directory path for error file.',
                        default='./badcase', type=str)

    args = parser.parse_args()
    return args



if __name__ == '__main__':
	args = parse_args()
	img_list = []

	with open(args.error_file) as f:
		lines = f.readlines()
		for line in lines:
			img_list.append(line.strip().split(",")[0])


	for img in img_list:
		shutil.copy(img, os.path.join(args.save_dir,os.path.basename(img)))

	
