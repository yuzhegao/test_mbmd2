"""
    Get name of all the training sequence
"""
import numpy as np
import glob
import os

img_root = '../../t_data/vid_data/Data/VID/train/'
mid_path = ['ILSVRC2015_VID_train_0000/','ILSVRC2015_VID_train_0001/']

output_file = 'vid_sequence_list.txt'

seq_list = []
for mid in mid_path:
    seqs = sorted(os.listdir(os.path.join(img_root,mid)))
    for i,seq in enumerate(seqs):
        seqs[i] = os.path.join(mid,seq)
    print len(seqs)
    seq_list = seq_list + seqs

print len(seq_list)
with open(output_file,'w') as f:
    for seq in seq_list:
        f.write(seq + '\n')



