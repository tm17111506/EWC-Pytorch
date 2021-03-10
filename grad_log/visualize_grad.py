from argparse import ArgumentParser
from pdb import set_trace as bp

import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser = ArgumentParser('EWC PyTorch Implementation')
parser.add_argument('--folder-path', type=str, default='/mnt/ssd2/docker/ubuntu-sshd/home/c_tma1/EWC-Pytorch/grad_log/02-02-21')
parser.add_argument('--num-task', type=int, default=3)
parser.add_argument('--log-interval', type=int, default=250)

if __name__ == '__main__':
	args = parser.parse_args()

	glob_min, glob_max, glob_start = 0, 0, False
	for i in range(args.num_task):
		task_folder = args.folder_path + os.sep + 'task_{}'.format(i+1)
		
		for file in os.listdir(task_folder):
			file_path = task_folder + os.sep + file
			
			if os.path.isfile(file_path):
				with open(file_path, 'rb') as f:
					d = pickle.load(f)
					vals = []
					for v in d.values():
						vals = vals + list(v.reshape(-1))

					curr_min = min(vals)
					curr_max = max(vals)

					if not glob_start:
						glob_start = True
						glob_min = curr_min
						glob_max = curr_max
					else:
						glob_min = min(glob_min, curr_min)
						glob_max = max(glob_max, curr_max)
	
	# Normalize values and store
	for i in range(args.num_task):
		task_folder = args.folder_path + os.sep + 'task_{}'.format(i+1)
		
		for file in os.listdir(task_folder):
			file_path = task_folder + os.sep + file
			
			if os.path.isfile(file_path):
				with open(file_path, 'rb') as f:
					d = pickle.load(f)

					norm_d = {k : (v - glob_min) / (glob_max - glob_min) for k,v in d.items()}

					for key, val in norm_d.items():
						img_nm = task_folder + os.sep + 'plt' + os.sep + os.path.splitext(file)[0] + '_' + key.replace('.', '_') + '.png'

						if len(val.shape) == 1 and val.shape[0] > 10:
							val = val.reshape(int(np.sqrt(val.shape[0])), -1)
						elif len(val.shape) == 1:
							val = val.reshape(-1, 1)

						val = val * 255.0
						val = val.astype(np.uint8)
						img = Image.fromarray(val)
						img.save(img_nm)

						# fig = plt.figure()
						# plt.matshow(val)
						# plt.colorbar()
						# fig.savefig(img_nm)
