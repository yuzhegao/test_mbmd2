"""
    Get training pair (template and img) from each seq
"""
import os
import pickle
import xml.etree.ElementTree as ET
import numpy as np

seq_home = '../vid_data/VID/Data/VID/train/'
ann_home = '../vid_data/VID/Annotations/VID/train/'

seq_list_file = 'vid_t.txt'

with open(seq_list_file) as f:
    lines = f.readlines()
    for i,line in enumerate(lines):
        lines[i] = line.rstrip()
seq_list = lines

sequences = {}
bbn = []
for i,seq in enumerate(seq_list):
	img_list = sorted([p for p in os.listdir(seq_home+seq) if os.path.splitext(p)[1] == '.JPEG'])
	ann_list = sorted([p for p in os.listdir(ann_home+seq) if os.path.splitext(p)[1] == '.xml'])
	bb = []
	cl = []
	drop_list = []
	print('seq %s, i %2d' % (seq, i))
	k = 0
	for j,ann in enumerate(ann_list):
		tree = ET.parse(ann_home+seq+'/'+ann)
		root = tree.getroot()
		try:
			#(min_x, min_y, max_x, max_y)
			bb.append([root[4][2][1].text, root[4][2][3].text
				, root[4][2][0].text, root[4][2][2].text])
		except IndexError:
			drop_list.append(j)
			img_list.pop(j-k)
			k += 1
	if drop_list != []:
		print('frame '+str(drop_list)+' of '+seq+" are dropped as it do not contain the target")
	assert len(img_list) == len(bb)
	sequences[seq] = {'images':img_list, 'bb':bb}
