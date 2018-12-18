import os
import sys
import numpy as np 
from PIL import Image

import torch
import torch.utils.data as data

sys.path.insert(0, '../modules')
from sample_generator import *
from utils import *

class get_dataset(data.Dataset):
	def __init__(self, img_dir, img_list, bb, configs):
		self.img_list = np.array([os.path.join(img_dir, img) for img in img_list])
		self.bb = np.array(bb)

		self.batch_frames = configs['batch_frames']
		self.frame_pos = configs['frame_pos']
		self.frame_neg = configs['frame_neg']
		self.scale_range = configs['scale_range']

		self.iou_pos = configs['iou_pos']
		self.iou_neg = configs['iou_neg']

		self.target_size = configs['target_size']
		self.padding = configs['padding']

		self.index = np.random.permutation(len(self.img_list))
		self.pointer = 0

		image = Image.open(self.img_list[0]).convert('RGB')
		#print(image.size)
		self.pos_generator = SampleGenerator('gaussian', image.size, 0.2, 1.2, 1.1, True)
		self.neg_generator = SampleGenerator('uniform', image.size, 1, 1.2, 1.1, True)

	def __iter__(self):
		return self

	def __next__(self):
		next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
		idx = self.index[self.pointer:next_pointer]
		if len(idx) < self.batch_frames:
			self.index = np.random.permutation(len(self.img_list))
			next_pointer = self.batch_frames - len(idx)
			idx = np.concatenate((idx, self.index[:next_pointer]))
		self.pointer = next_pointer

		pos_bb = []
		neg_bb = []
		images = []
		#print(self.img_list[idx])
		#print(self.bb[idx])
		for i, (img_path, bb) in enumerate(zip(self.img_list[idx], self.bb[idx])):
			image = Image.open(img_path).convert('RGB')
			image = np.asarray(image)
			#( 32, 4) ( 96, 4)
			pos_samples = gen_samples(self.pos_generator, bb, self.frame_pos, iou_range=self.iou_pos)
			neg_samples = gen_samples(self.neg_generator, bb, self.frame_neg, iou_range=self.iou_neg)
			samples = np.concatenate((pos_samples, neg_samples), axis=0)
			#get the bbox that enclose all the samples
			enclosing_bb = [np.min(samples[:][0]), np.min(samples[:][1]),
							np.max(samples[:][2]), np.max(samples[:][3])]
			image, pos_samples, neg_samples = resize_crop(self.target_size, image, bb, enclosing_bb, pos_samples, neg_samples)
			image = torch.from_numpy(image.T).float().unsqueeze(dim=0)
			pos_bb.append(pos_samples)
			neg_bb.append(neg_samples)
			images.append(image)
			
			
		pos_samples = torch.from_numpy(np.concatenate(pos_bb, axis=0)).float() / float(8)
		neg_samples = torch.from_numpy(np.concatenate(neg_bb, axis=0)).float() / float(8)
		# images are cropped and resized
		# pos_samples(32, 4), neg_samples(96, 4)(min_x, min_y, max_x, max_y) are coordinates of samples 
		# on feature map(conv3) 
		return images, pos_samples, neg_samples
	next = __next__


		
