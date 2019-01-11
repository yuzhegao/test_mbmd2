import os
import sys
import pickle
import random
import numpy as np

sys.path.append('../')

import torch
import torch.utils.data as data

import cv2
from PIL import Image
from train.data_utils import draw_box

def crop_search_region(img, gt, win_size, scale=4, mean_rgb=128, offset=None):
    ## crop the search region, which is four times the size of the target,and centering in gt's center
    # gt: [ymin, xmin, ymax, xmax]

    bnd_ymin, bnd_xmin, bnd_ymax, bnd_xmax = gt
    bnd_w = bnd_xmax - bnd_xmin
    bnd_h = bnd_ymax - bnd_ymin
    # cx, cy = gt[:2] + gt[2:] / 2
    cy, cx = (bnd_ymin + bnd_ymax)/2, (bnd_xmin+bnd_xmax)/2
    #diag = np.sum(bnd_h** 2 + bnd_w**2) ** 0.5
    #origin_win_size = diag * scale
    origin_win_size_h, origin_win_size_w = bnd_h * scale, bnd_w * scale
    # origin_win_size_h = origin_win_size
    # origin_win_size_w = origin_win_size
    #print "che {} {}".format(img.size,img.size[1::-1]) ##(576, 432) (432, 576)
    im_size = img.size[1::-1] ##[H,W]
    min_x = np.round(cx - origin_win_size_w / 2).astype(np.int32)
    max_x = np.round(cx + origin_win_size_w / 2).astype(np.int32)
    min_y = np.round(cy - origin_win_size_h / 2).astype(np.int32)
    max_y = np.round(cy + origin_win_size_h / 2).astype(np.int32)

    if offset is not None:
        min_offset_y, max_offset_y = (bnd_ymax - max_y, bnd_ymin - min_y)
        min_offset_x, max_offset_x = (bnd_xmax - max_x, bnd_xmin - min_x)
        offset[0] = np.clip(offset[0] * origin_win_size_h, min_offset_y, max_offset_y)
        offset[1] = np.clip(offset[1] * origin_win_size_w, min_offset_x, max_offset_x)
        offset = np.int32(offset)
        min_y += offset[0]
        max_y += offset[0]
        min_x += offset[1]
        max_x += offset[1]

    #print "che {} / {}".format(gt, [min_y,min_x,max_y,max_x])
    ## in fact, the [min_y,min_x,max_y,max_x] always out of image

    win_loc = np.array([min_y, min_x]) ## what if min_x/min_y <0 ???

    gt_x_min, gt_y_min = ((bnd_xmin-min_x)/origin_win_size_w, (bnd_ymin - min_y)/origin_win_size_h)
    gt_x_max, gt_y_max = [(bnd_xmax-min_x)/origin_win_size_w, (bnd_ymax - min_y)/origin_win_size_h]
    # coordinates on window
    #relative coordinates of gt bbox to the search region

    unscaled_w, unscaled_h = [max_x - min_x + 1, max_y - min_y + 1] ## before scaled to 300*300
    min_x_win, min_y_win, max_x_win, max_y_win = (0, 0, unscaled_w, unscaled_h)
    ## in search region coordinate

    min_x_im, min_y_im, max_x_im, max_y_im = (min_x, min_y, max_x+1, max_y+1)
    ## in origin img coordinate   (useless)

    img = img.crop([min_x_im, min_y_im, max_x_im, max_y_im]) ## crop the search region
    ## from the code below: if the min/max out of origin img bound, then just padding
    img_array = np.array(img)

    if min_x < 0:
        min_x_im = 0
        min_x_win = 0 - min_x
    if min_y < 0:
        min_y_im = 0
        min_y_win = 0 - min_y
    if max_x+1 > im_size[1]:
        max_x_im = im_size[1]
        max_x_win = unscaled_w - (max_x + 1 - im_size[1])
    if max_y+1 > im_size[0]:
        max_y_im = im_size[0]
        max_y_win = unscaled_h - (max_y + 1 - im_size[0])
    ## after padding

    unscaled_win = np.ones([unscaled_h, unscaled_w, 3], dtype=np.uint8) * np.uint8(mean_rgb)
    unscaled_win[min_y_win:max_y_win, min_x_win:max_x_win] = img_array[min_y_win:max_y_win, min_x_win:max_x_win]
    ## here padding with 128(mean value)

    unscaled_win = Image.fromarray(unscaled_win)
    height_scale, width_scale = np.float32(unscaled_h)/win_size, np.float32(unscaled_w)/win_size
    win = unscaled_win.resize([win_size, win_size], resample=Image.BILINEAR)

    # win = sp.misc.imresize(unscaled_win, [win_size, win_size])
    return win, np.array([gt_y_min, gt_x_min, gt_y_max, gt_x_max]), win_loc, [height_scale, width_scale]

class vid_dataset(data.Dataset):
    def __init__(self,pair_file,img_root,template_size = 128,img_size = 300):
        with open(pair_file, 'rb') as fp:
            data = pickle.load(fp)
        self.pairs = data
        self.template_size = template_size
        self.img_size = img_size
        self.img_root = img_root

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        template_file,img_file = pair[0],pair[1]
        temp_gt,img_gt = np.array(pair[2]).astype(np.float32),np.array(pair[3]).astype(np.float32) ## x1 y1 x2 y2
        init_gt = np.array([temp_gt[1],temp_gt[0],temp_gt[3],temp_gt[2]])
        cur_gt = np.array([img_gt[1],img_gt[0],img_gt[3],img_gt[2]]) ## y1 x1 y2 x2

    ## prepare for template
        init_img_array = cv2.imread(self.img_root + template_file)
        init_img = Image.fromarray(init_img_array)  ## the first frame
        init_img_array = np.array(init_img)
        if init_img_array.ndim < 3:
            init_img_array = np.expand_dims(init_img_array, axis=2)
            init_img_array = np.repeat(init_img_array, repeats=3, axis=2)
            init_img = Image.fromarray(init_img_array)

        Padding = True
        init_img_width, init_img_height = init_img.size

        img1_xiaobai = np.array(init_img)
        gt_boxes = np.zeros((1, 4))
        gt_boxes[0, 0] = init_gt[0] / float(init_img_height)
        gt_boxes[0, 1] = init_gt[1] / float(init_img_width)
        gt_boxes[0, 2] = init_gt[2] / float(init_img_height)
        gt_boxes[0, 3] = init_gt[3] / float(init_img_width)

        if Padding:
            pad_x = 36.0 / 264.0 * (gt_boxes[0, 3] - gt_boxes[0, 1]) * init_img_width
            pad_y = 36.0 / 264.0 * (gt_boxes[0, 2] - gt_boxes[0, 0]) * init_img_height
            startx = gt_boxes[0, 1] * init_img_width - pad_x
            starty = gt_boxes[0, 0] * init_img_height - pad_y
            endx = gt_boxes[0, 3] * init_img_width + pad_x
            endy = gt_boxes[0, 2] * init_img_height + pad_y

            left_pad = max(0, int(-startx))
            top_pad = max(0, int(-starty))
            right_pad = max(0, int(endx - init_img_width + 1))
            bottom_pad = max(0, int(endy - init_img_height + 1))  ## prevent bbox out of init_img

            ## re-compute the x1,x2,y1,y2 after padding
            startx = int(startx + left_pad)
            starty = int(starty + top_pad)
            endx = int(endx + left_pad)
            endy = int(endy + top_pad)

            if top_pad or left_pad or bottom_pad or right_pad:
                r = np.pad(img1_xiaobai[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                           constant_values=128)
                g = np.pad(img1_xiaobai[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                           constant_values=128)
                b = np.pad(img1_xiaobai[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                           constant_values=128)  ## padding value=128 ?
                r = np.expand_dims(r, 2)
                g = np.expand_dims(g, 2)
                b = np.expand_dims(b, 2)

                img1_xiaobai = np.concatenate((r, g, b), axis=2)
        else:
            startx = gt_boxes[0, 1] * init_img_width
            starty = gt_boxes[0, 0] * init_img_height
            endx = gt_boxes[0, 3] * init_img_width
            endy = gt_boxes[0, 2] * init_img_height

        img1_xiaobai = Image.fromarray(img1_xiaobai)

        # gt_boxes resize
        init_img_crop = img1_xiaobai.crop(np.int32([startx, starty, endx, endy]))
        init_img_crop = init_img_crop.resize([128, 128], resample=Image.BILINEAR)

        init_img_array = np.array(init_img_crop)



    ## prepare for search region
        img = cv2.imread(self.img_root + img_file)
        img = Image.fromarray(img)

        offset_x = random.random() - 0.5
        offset_y = random.random() - 0.5
        scale = np.clip(random.random()*4,2,4)
        search_region, _, win_loc, scaled = crop_search_region(img, cur_gt,
                                                    self.img_size,scale=scale, offset=[offset_x, offset_y])

        gt_search_region = [(cur_gt[0] - win_loc[0]) / scaled[0],
                            (cur_gt[1] - win_loc[1]) / scaled[1],
                            (cur_gt[2] - win_loc[0]) / scaled[0],
                            (cur_gt[3] - win_loc[1]) / scaled[1]] ## y1 x1 y2 x2

        template = init_img_array  ## (128,128,3)
        seq_input_img = search_region  ## (300,300,3)
        seq_input_gt = np.array(gt_search_region)  ## (4,)

        return template, seq_input_gt, seq_input_img


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

    img_batch = np.stack(img_batch, axis=0)  ## (B,300,300,3)
    gt_batch = np.stack(gt_batch, axis=0) / 300.0
    gt_batch = np.expand_dims(gt_batch, axis=1)  ## (B,1,4)
    template_batch = np.stack(template_batch, axis=0)  ## (B,128,128,3)

    batchsize, _, _, _ = img_batch.shape
    label_batch = np.ones((batchsize, 1, 1), dtype=np.int)  ## (B,1,1)
    ## when use np.zeros, the location_loss = 0
    ## so use np.ones

    return template_batch, img_batch, gt_batch, label_batch


################################################
if __name__ == '__main__':
    test_loader = vid_dataset('data_prepare/vid_all.pkl','../t_data/vid_data/Data/VID/train/')
    print (len(test_loader))
    data_loader = torch.utils.data.DataLoader(test_loader,batch_size=4,
                                              shuffle=True, num_workers=1, collate_fn=vid_collate)
    # a = test_loader[7]
    # print (len(test_loader))

    for idx, (templates, imgs, gts, labels) in enumerate(data_loader):
        print('tmps ',templates.shape)
        print('imgs ',imgs.shape)
        print('gts ',gts.shape) ## y1 x1 y2 x2
        print('labels ',labels.shape)
        print(imgs[3][0].dtype)

        print idx
        img1 = Image.fromarray(imgs[0])
        gt1 = gts[0][0]*300
        gt1 = [gt1[1],gt1[0],gt1[3],gt1[2]]## x1 y1 x2 y2
        draw_box(img1, gt1, img_path='tmp2/{}.jpg'.format(idx))
        
        #print (imgs.shape)
        print (templates.dtype)
