import os
import pickle
from collections import OrderedDict
import xml.etree.ElementTree as ET 

seq_home = '../dataset/VID/Data/VID/train/ILSVRC2015_VID_train_0000/'
ann_home = '../dataset/VID/Annotations/VID/train/ILSVRC2015_VID_train_0000/'

seq_list = sorted([p for p in os.listdir(seq_home)])

for i,seq in enumerate(seq_list):
	#img_list = sorted([p for p in os.listdir(seq_home+seq) if os.path.splitext(p)[1] == '.JPEG'])
	ann_list = sorted([p for p in os.listdir(ann_home+seq) if os.path.splitext(p)[1] == '.xml'])
	fp = open(seq_home+seq+'/'+'bb.txt', 'w') 
	for j,ann in enumerate(ann_list):
		tree = ET.parse(ann_home+seq+'/'+ann)
		root = tree.getroot()
		try:
			
			fp.write(str(root[4][2][1].text)+','+'\t')
			fp.write(str(root[4][2][2].text)+','+'\t')
			fp.write(str(int(root[4][2][0].text) - int(root[4][2][1].text))+','+'\t')
			fp.write(str(int(root[4][2][2].text) - int(root[4][2][3].text))+','+'\t')
			fp.write('\n')
		except IndexError:
			#open(seq_home+seq+'/'+'bb.txt', 'w') 
			fp.write(str(0)+','+'\t')
			fp.write(str(0)+','+'\t')
			fp.write(str(0)+','+'\t')
			fp.write(str(0)+','+'\t')
			fp.write('\n')
			
	fp.close()

