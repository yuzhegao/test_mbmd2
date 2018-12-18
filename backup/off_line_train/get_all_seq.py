"""
    Get name of all the training sequence
"""
import numpy as np
import glob
import os

img_root = '../../t_data/vid_data/Data/'
mid_path = ['VID/train/ILSVRC2015_VID_train_0000/','VID/train/ILSVRC2015_VID_train_0001/']

output_file = 'vid_sequence_list.txt'

seq_list = []
for mid in mid_path:
    seqs = sorted(os.listdir(os.path.join(img_root,mid)))
    for i,seq in enumerate(seqs):
        seqs[i] = os.path.join(mid_path,seq)
    seq_list = seq_list + seqs

# print sorted(os.listdir('/home/yuzhe/Downloads/vot2018_sequence/sequences'))
# print sorted(glob.glob('/home/yuzhe/Downloads/vot2018_sequence/sequences/*'))

with open(output_file,'r') as f:
    for seq in seq_list:
        f.write(seq + '\n')
