import os
import sys
import pickle
import time

import torch
import torch.optim as optim
import torch.tensor

from get_dataset import *
from model import *
from configs import *

img_home = '../dataset/VID/Data/VID/train/'
seq_path = 'sequences/vid.pkl'

def set_optimizer(model, lr_base, lr_mult=configs['lr_mult'], momentum=configs['momentum'], w_decay=configs['w_decay']):
	params = model.get_learnable_params()
	param_list = []
	for k, p in params.items():
		lr = lr_base
		for l, m in lr_mult.items():
			if k.startswith(l):
				lr = lr_base * m 
		param_list.append({'params': [p], 'lr':lr})
	optimizer = optim.SGD(param_list, lr = lr, momentum=momentum, weight_decay=w_decay)
	return optimizer


def offline_train():
	## Init dataset ##
	with open(seq_path, 'rb') as fp:
		data = pickle.load(fp)

	K = len(data)
	print(K)
	dataset = [None] * K
	for k, (seqname, seq) in enumerate(data.items()):
		print('seq %s k %2d' % (seqname, k))
		img_list = seq['images']
		bb = np.asarray(seq['bb'], dtype=int)
		img_dir = os.path.join(img_home, seqname)
		dataset[k] = get_dataset(img_dir, img_list, bb, configs)
	## Init model ##
	model = RT_MDNet(configs['vggm_weight_path'], K)	
	if configs['use_gpu']:
		#print(torch.cuda.device_count())
		model = model.cuda()
	model.set_learnable_params(configs['ft_layers'])
	
	## Init criterion and optimizer ##
	criterion = MultiTaskLoss()
	evaluator_p = Precision()
	evaluator_a = Accuracy()
	optimizer = set_optimizer(model, configs['lr'])

	best_prec = 0.
	for i in range(configs['n_cycles']):
		print("=== Start Cycle %d ===" % (i))
		k_list = np.random.permutation(K)
		cls_prec = np.zeros(K)
		ins_prec = np.zeros(K)
		pos_acc = np.zeros(K)
		neg_acc = np.zeros(K)
		for j,k in enumerate(k_list):
			tic = time.time()
			images, pos_regions, neg_regions = dataset[k].next()

			if configs['use_gpu']:
				images = images[0].cuda()
				pos_regions = pos_regions.cuda()
				neg_regions = neg_regions.cuda()
			if len(neg_regions.cpu().numpy()) == 0:
				continue
			try:
				pos_score = model(images, pos_regions, k, pos=True)
				neg_score = model(images, neg_regions, k)
			except RuntimeError:
				print('images shape', images.shape)
				print('pos_regions', pos_regions)
				print('neg_regions', neg_regions.cpu().numpy()) 
				continue

			loss, cls_loss = criterion(pos_score, neg_score, k)
			model.zero_grad()
			loss.backward()
			#torch.nn.utils.clip_grad_norm(model.parameters(), configs['grad_clip'])
			optimizer.step()

			cls_prec[k], ins_prec[k] = evaluator_p(pos_score, neg_score, k)
			pos_acc[k], neg_acc[k] = evaluator_a(pos_score, neg_score, k)
			
			print('Cycle %2d, K %2d (%2d), Loss %.3f, cls_loss %.3f, Cls_Prec %.3f, Ins_Prec %.3f, Pos_Acc %.3f, Neg_Acc %.3f' % \
				(i, j, k, loss.data, cls_loss.data, cls_prec[k], ins_prec[k], pos_acc[k], neg_acc[k]))
		if (i%100) == 0:
			torch.save(model.state_dict(), configs['trained_model_path'])
		cls_prec = cls_prec.mean()
		ins_prec = ins_prec.mean()
		pos_acc = pos_acc.mean()
		neg_acc = neg_acc.mean()
		print('Cycle %2d, cls_prec %.3f, ins_prec %.3f, pos_acc %.3f, neg_acc %.3f' % \
			(i, cls_prec, ins_prec, pos_acc, neg_acc))
				
				
	
			





if __name__ == "__main__":
	offline_train()

