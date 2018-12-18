import os
import pickle
import numpy as np

import torch
import torch.utils.data as data

from PIL import Image
from train.data_utils import crop_search_region

class vid_dataset(data.Dataset):
    def __init__(self,pair_file,template_size = 128,img_size = 300):
        with open(pair_file, 'rb') as fp:
            data = pickle.load(fp)
        self.pairs = data
        self.template_size = template_size
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        template_file,img_file = pair[0],pair[1]
        temp_gt,img_gt = np.array(pair[2]).astype(np.float32),np.array(pair[3]).astype(np.float32) ## x1 y1 x2 y2

        template,img = Image.open(template_file),Image.open(img_file)
        if not (np.array(img).ndim) == 3:
            img_array = np.expand_dims(np.array(img), axis=2)
            img_array = np.tile(img_array, (1, 1, 3))
            img = Image.fromarray(img_array)
        if not (np.array(template).ndim) == 3:
            temp_array = np.expand_dims(np.array(template), axis=2)
            temp_array = np.tile(temp_array, (1, 1, 3))
            template = Image.fromarray(temp_array)

        template = template.crop(temp_gt)
        template = template.resize([self.template_size, self.template_size])

        search_region, win_loc, scaled = crop_search_region(img, img_gt, self.img_size)

        #template,search_region = np.array(template),np.array(search_region)
        gt_search_region = [(img_gt[0] - win_loc[0]) / scaled[0],
                            (img_gt[1] - win_loc[1]) / scaled[1],
                            (img_gt[2] - win_loc[0]) / scaled[0],
                            (img_gt[3] - win_loc[1]) / scaled[1]]  ##(x1,y1,x2,y2)
        new_gt = [gt_search_region[1], gt_search_region[0], gt_search_region[3], gt_search_region[2]]
        ## (y1,x1,y2,x2)

        template = np.array(template)  ##transpose to (h,w,3)
        input_img = np.expand_dims(np.array(search_region), axis=0)
        input_gt = np.expand_dims(new_gt, axis=0)

        return template, input_gt, input_img


def vid_collate(batch):
    """
     batch: a list of (template: (ht,wt,3)
                        seq_input_gt,:  (num_seq,4)
                        seq_input_img:  (num_seq,h,w,3)
                        )
    """
    img_batch = []
    gt_batch = []
    template_batch = []

    for sample in batch:
        template_batch.append(sample[0])
        img_batch.append(sample[2])
        gt_batch.append(sample[1])

    img_batch = np.stack(img_batch, axis=0)
    gt_batch = np.stack(gt_batch, axis=0)/300.0
    template_batch = np.expand_dims(np.stack(template_batch,axis=0),axis=1)
    batchsize,num_seq,_ = gt_batch.shape

    label_batch = np.ones((batchsize,num_seq,1),dtype=np.int)
    ## when use np.zeros, the location_loss = 0
    ## so use np.ones

    return template_batch,img_batch,gt_batch,label_batch


################################################
if __name__ == '__main__':
    test_loader = vid_dataset('/home/yuzhe/Downloads/part_vot_seq/')
    print (len(test_loader))
    data_loader = torch.utils.data.DataLoader(test_loader,batch_size=4, shuffle=False, num_workers=1, collate_fn=vid_collate)
    # a = test_loader[7]
    # print (len(test_loader))

    for idx, (templates, imgs, gts, labels) in enumerate(data_loader):
        # print(templates.shape)
        print (imgs.shape)
        # print(gts.shape)
        # print(labels.shape)
        # print(imgs[3][0].dtype)
        # img1 = Image.fromarray(imgs[0][0])
        # gt1 = gts[0][0]
        # gt1 = [gt1[1],gt1[0],gt1[3],gt1[2]]## (y1,x1,y2,x2)
        # draw_box(img1, gt1, img_path='/home/yuzhe/tmp/{}.jpg'.format(idx))
        print (templates.shape)
        print (templates.dtype)
