from __future__ import print_function,division

import os
import cv2
import glob
import numpy as np
import random
from PIL import Image
from data_utils import draw_box,transform

import torch
import torch.utils.data as data

def convert_to_one_hot(y, C):
    y = np.array(y)
    return np.eye(C)[y.reshape(-1)]

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
    # return win, np.array([gt_x_min, gt_y_min, gt_x_max, gt_y_max]), diag, np.array(win_loc)


## otb raw gt : [x y w h] & (x,y) in low corner
class basketball_dataset(data.Dataset):
    def __init__(self,data_rootpath, num_seq=1, frame_interval=2, template_size=128, img_size=300):
        self.data_rootpath = data_rootpath
        self.num_seq = 1
        self.template_size = template_size
        self.img_size = img_size
        self.frame_interval = frame_interval

        self.seq_list = sorted(glob.glob(data_rootpath + '/*'))
        for i, seq in enumerate(self.seq_list):
            self.seq_list[i] = os.path.basename(seq)

        # print seq_list
        self.seq_img_dict = dict()
        self.gt_dict = dict()
        for seq in self.seq_list:
            #print seq
            img_list = sorted(glob.glob(os.path.join(self.data_rootpath, seq) + '/img/*.jpg'))
            self.seq_img_dict[seq] = img_list

            with open(os.path.join(self.data_rootpath, seq, 'groundtruth_rect.txt'), 'r') as fl:
                lines = fl.readlines()

                if len(lines[0].rstrip().split(',')) > 1:
                    for i, line in enumerate(lines):
                        lines[i] = line.rstrip().split(',')
                else:
                    for i, line in enumerate(lines):
                        lines[i] = line.rstrip().split()

                lines = np.array(lines, dtype=np.float32).astype(np.int) ## (x y w h)
                lines[:,2] = lines[:,0] + lines[:,2]
                lines[:,3] = lines[:,1] + lines[:,3] ##(x1,y1,x2,y2)

                gt_seq = np.zeros_like(lines)
                gt_seq[:, 0] = lines[:, 1]
                gt_seq[:, 1] = lines[:, 0]
                gt_seq[:, 2] = lines[:, 3]
                gt_seq[:, 3] = lines[:, 2]  ## (y1 x1 y2 x2)

                self.gt_dict[seq] = gt_seq

        self.basketball_imgs = self.seq_img_dict['Basketball']
        self.basketball_gts = self.gt_dict['Basketball']
        assert len(self.basketball_imgs) == len(self.basketball_gts)

    def __len__(self):
        return len(self.basketball_imgs) - self.frame_interval - 3

    def draw_gtbox(self,img,gt,path):
        x1, y1, x2, y2 = gt[1],gt[0],gt[3],gt[2]
        gt = [x1,y1,x2,y2]
        draw_box(img,gt,img_path=path)

    def __getitem__(self, idx):
        template_file = self.basketball_imgs[0]
        img_file = self.basketball_imgs[idx + self.frame_interval]
        # print (template_file,img_file)
        init_gt = self.basketball_gts[0]
        cur_gt = self.basketball_gts[idx + self.frame_interval]

        # init_img_array = cv2.imread(template_file)
        # init_img = Image.fromarray(init_img_array)
        # self.draw_gtbox(init_img,init_gt,'../tmp3/img1.jpg')
        #
        # img = cv2.imread(img_file)
        # img = Image.fromarray(img)
        # self.draw_gtbox(img, cur_gt, '../tmp3/img2.jpg')

        #------------------------
        # prepare template (some padding)
        init_img_array = cv2.imread(template_file)

        init_img = Image.fromarray(init_img_array)  ## the first frame
        init_img_array = np.array(init_img)
        if init_img_array.ndim < 3:
            init_img_array = np.expand_dims(init_img_array, axis=2)
            init_img_array = np.repeat(init_img_array, repeats=3, axis=2)
            init_img = Image.fromarray(init_img_array)

        ## init_gt  # ymin xmin ymax xmax

        ## if not padding, the size of predict box will not be very correct
        Padding = True
        init_img_width,init_img_height = init_img.size

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

        #------------------------

        img = cv2.imread(img_file)
        img = Image.fromarray(img)

        offset_x = random.random() - 0.5
        offset_y = random.random() - 0.5
        search_region,_, win_loc, scaled = crop_search_region(img, cur_gt, self.img_size,offset=[offset_x,offset_y])

        gt_search_region = [(cur_gt[0] - win_loc[0]) / scaled[0],
                            (cur_gt[1] - win_loc[1]) / scaled[1],
                            (cur_gt[2] - win_loc[0]) / scaled[0],
                            (cur_gt[3] - win_loc[1]) / scaled[1]]  ## y1 x1 y2 x2



        template = init_img_array  ## (128,128,3)
        seq_input_img = search_region ## (300,300,3)
        seq_input_gt = np.array(gt_search_region) ## (4,)

        return template, seq_input_gt, seq_input_img


def otb_collate(batch):
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

    img_batch = np.stack(img_batch, axis=0) ## (B,300,300,3)
    gt_batch = np.stack(gt_batch, axis=0)/300.0
    gt_batch = np.expand_dims(gt_batch,axis=1)  ## (B,1,4)
    template_batch = np.stack(template_batch,axis=0) ## (B,128,128,3)

    batchsize,_,_,_ = img_batch.shape
    label_batch = np.ones((batchsize,1,1),dtype=np.int) ## (B,1,1)
    ## when use np.zeros, the location_loss = 0
    ## so use np.ones

    return template_batch,img_batch,gt_batch,label_batch

########################################################################



if __name__ == '__main__':
    test_loader = basketball_dataset('/home/yuzhe/Downloads/part_vot_seq/')
    print (len(test_loader))
    data_loader = torch.utils.data.DataLoader(test_loader,
                                              batch_size=4, shuffle=False, num_workers=1, collate_fn=otb_collate)
    a = test_loader[7]
    print (len(test_loader))

    # for idx, (templates, imgs, gts, labels) in enumerate(data_loader):
    #     print (idx,'\n')
    #     print(templates.shape)
    #     print (imgs.shape)
    #     print(gts.shape)
    #     print(labels.shape)
    #     # print(imgs[3][0].dtype)
    #     # img1 = Image.fromarray(imgs[0][0])
    #     # gt1 = gts[0][0]
    #     # gt1 = [gt1[1],gt1[0],gt1[3],gt1[2]]## (y1,x1,y2,x2)
    #     #draw_box(img1, gt1, img_path='/home/yuzhe/tmp/{}.jpg'.format(idx))
    #     print (templates.dtype)
    #
    #
    #     # seq = test_loader.seq_list[10]
    #     # num_seq = len(test_loader.label_dict['Biker'])
    #     # for i in range(num_seq):
    #     #    test_loader.draw_gtbox('Biker', i)



