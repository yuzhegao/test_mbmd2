import os
import sys
import pickle
import time

import torch
import torch.optim as optim
import torch.tensor

from get_dataset import *

img_home = '../dataset/VID/Data/VID/train/'
seq_path = 'sequences/vid.pkl'

with open(seq_path, 'rb') as fp:
	data = pickle.load(fp)

	K = len(data)
	print K

	dataset = [None] * K
	for k, (seqname, seq) in enumerate(data.items()):
		# print seqname
		# print seq
		img_list = seq['images']
		bb = seq['bb']
		img_dir = os.path.join(img_home, seqname)
		print img_dir
		print img_list


		#dataset[k] = get_dataset(img_dir, img_list, bb, configs)

	