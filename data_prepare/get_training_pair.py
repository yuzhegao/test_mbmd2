import os
import sys
import pickle
import xml.etree.ElementTree as ET
import numpy as np
import glob
from PIL import Image, ImageOps, ImageStat, ImageDraw

sys.path.append('../')
sys.path.append('../train/')

from train.data_utils import draw_box,draw_mulitbox,iou_y1x1y2x2,crop_search_region

seq_home = '../../t_data/vid_data/Data/VID/train/'
ann_home = '../../t_data/vid_data/Annotations/VID/train/'
output_path = 'vid_all.pkl'

seq_list_file = 'vid_sequence_list.txt'
num_interval = 3

with open(seq_list_file) as f:
    lines = f.readlines()
    for i,line in enumerate(lines):
        lines[i] = line.rstrip()
#for line in lines:
#    print line

seq_list = lines
pair_list = []
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
            bb.append([root[4][2][1].text,root[4][2][3].text,
                        root[4][2][0].text,root[4][2][2].text]) ## x1 y1 x2 y2
        except IndexError:
            drop_list.append(j)
            img_list.pop(j-k)
            k += 1
    if drop_list != []:
        print('frame '+str(drop_list)+' of '+seq+" are dropped as it do not contain the target")
    print len(img_list), len(bb)
    assert len(img_list) == len(bb)

    len_seq = len(bb)
    num_train_pair = len_seq - num_interval - 2
    pair_list_seq = []
    for k in xrange(num_train_pair):
        pair_list_seq.append([os.path.join(seq,img_list[k]),
                              os.path.join(seq,img_list[k+num_interval]),
                              bb[k],
                              bb[k+num_interval]])
    pair_list = pair_list + pair_list_seq
    print len(pair_list)


def check_sample(idx):
    sample1 = pair_list[idx]
    temp = sample1[0]
    img = sample1[1]
    temp_gt,img_gt = sample1[2],sample1[3]
    print temp_gt
    draw_box(Image.open(seq_home + temp),np.array(temp_gt).astype(np.float32),'tmp/{}_temp.jpg'.format(idx))
    #draw_box(Image.open(seq_home + img),np.array(img_gt).astype(np.float32),'tmp/{}_img.jpg'.format(idx))

with open(output_path, 'wb') as fp:
    pickle.dump(pair_list, fp, -1)

#for n in range(0,2530):
#    check_sample(n)

