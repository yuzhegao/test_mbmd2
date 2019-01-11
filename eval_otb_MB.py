# -- coding: utf-8 --

import cv2
import os
import sys
sys.path.append('./lib')
sys.path.append('./lib/slim')

from lib.object_detection.protos import pipeline_pb2
from lib.object_detection.core import box_list
from lib.object_detection.core import box_list_ops
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import vot
from vggm import vggM
from sample_generator import *
from tracking_utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="0"





def get_configs_from_pipeline_file(config_file):
  """Reads training configuration from a pipeline_pb2.TrainEvalPipelineConfig.

  Reads training config from file specified by pipeline_config_path flag.

  Returns:
    model_config: model_pb2.DetectionModel
    train_config: train_pb2.TrainConfig
    input_config: input_reader_pb2.InputReader
  """
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(config_file, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  model_config = pipeline_config.model.ssd
  train_config = pipeline_config.train_config
  input_config = pipeline_config.train_input_reader
  eval_config = pipeline_config.eval_config

  return model_config, train_config, input_config, eval_config


def restore_model(sess, model_scope, checkpoint_path, variables_to_restore):
    # variables_to_restore = tf.global_variables()
    name_to_var_dict = dict([(var.op.name.lstrip(model_scope+'/'), var) for var in variables_to_restore
                             if not var.op.name.endswith('Momentum')])
    saver = tf.train.Saver(name_to_var_dict)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, latest_checkpoint)

def crop_search_region(img, gt, win_size, scale=4, crop_size=None, mean_rgb=128, offset=None):
    ## crop the search region, which is four times the size of the target,and centering in gt's center
    # gt: [ymin, xmin, ymax, xmax]

    bnd_ymin, bnd_xmin, bnd_ymax, bnd_xmax = gt
    bnd_w = bnd_xmax - bnd_xmin
    bnd_h = bnd_ymax - bnd_ymin
    # cx, cy = gt[:2] + gt[2:] / 2
    cy, cx = (bnd_ymin + bnd_ymax)/2, (bnd_xmin+bnd_xmax)/2
    #diag = np.sum(bnd_h** 2 + bnd_w**2) ** 0.5
    #origin_win_size = diag * scale
    if crop_size is None:
        origin_win_size_h, origin_win_size_w = bnd_h * scale, bnd_w * scale
    else:
        origin_win_size_h, origin_win_size_w = crop_size[0] * scale, crop_size[1] * scale
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
    ## now　resize and get "resize_scale_rate"

    # win = sp.misc.imresize(unscaled_win, [win_size, win_size])
    return win, np.array([gt_y_min, gt_x_min, gt_y_max, gt_x_max]), win_loc, [height_scale, width_scale]
    # return win, np.array([gt_x_min, gt_y_min, gt_x_max, gt_y_max]), diag, np.array(win_loc)



class MobileTracker(object):
    def __init__(self, image, region):
        init_training = True
        config_file = 'MBMD_vot_model/model/ssd_mobilenet_video.config'
        # checkpoint_dir = '/home/yuzhe/PycharmProjects/test_mbmd/model/ssd_mobilenet_video1/'
        # checkpoint_dir = '/home/yuzhe/vid_model_1/'
        checkpoint_dir = 'MBMD_vot_model/model/dump'

        model_config, train_config, input_config, eval_config = get_configs_from_pipeline_file(config_file)
        model = build_man_model(model_config=model_config, is_training=False)
        ## fn build_man_model: return a MANMetaArch netwrk object

        model_scope = 'model'

        ##-------------------------------------------------------------------------------------------------

        self.initFeatOp, self.initInputOp = build_init_graph(model, model_scope, reuse=None)
        ## self.initFeatOp -> template feature vector
        ## self.initFeatOp :using mobilenet process 127*127 template patch

        self.initConstantOp = tf.placeholder(tf.float32, [1,1,1,512]) ## maybe the feature map of  template
        ## notice: one-shot learning !

        ## maybe:  build_box_predictor -> using siameseRPN to predict proposal box
        self.pre_box_tensor, self.scores_tensor, self.input_cur_image = build_box_predictor(model, model_scope, self.initConstantOp, reuse=None)
        ## predict bbox

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #if not init_training:
        variables_to_restore = tf.global_variables()

        """
        for i in variables_to_restore:
            print i
        exit()
        """

        restore_model(self.sess, model_scope, checkpoint_dir, variables_to_restore) ## restore the siamese network checkpoint

        ##-------------------------------------------------------------------------------------------------
        ## prepare for the template feature vector

        init_img = Image.fromarray(image) ## the first frame
        init_img_array = np.array(init_img)
        self.expand_channel = False
        if init_img_array.ndim < 3:
            init_img_array = np.expand_dims(init_img_array, axis=2)
            init_img_array = np.repeat(init_img_array, repeats=3, axis=2)
            init_img = Image.fromarray(init_img_array)
            self.expand_channel = True

        init_gt1 = [region.x,region.y,region.width,region.height]
        init_gt = [init_gt1[1], init_gt1[0], init_gt1[1]+init_gt1[3], init_gt1[0]+init_gt1[2]] # ymin xmin ymax xmax


        init_img_width,init_img_height = init_img.size

        gt_boxes = np.zeros((1,4))
        gt_boxes[0,0] = init_gt[0] / float(init_img_height)
        gt_boxes[0,1] = init_gt[1] / float(init_img_width)
        gt_boxes[0,2] = init_gt[2] / float(init_img_height)
        gt_boxes[0,3] = init_gt[3] / float(init_img_width)  ## ymin xmin ymax xmax  -> relative coord to origin img  (very confused )

        img1_xiaobai = np.array(init_img)
        pad_x = 36.0 / 264.0 * (gt_boxes[0, 3] - gt_boxes[0, 1]) * init_img_width
        pad_y = 36.0 / 264.0 * (gt_boxes[0, 2] - gt_boxes[0, 0]) * init_img_height ## ???
        #cx = (gt_boxes[0, 3] + gt_boxes[0, 1]) / 2.0 * init_img.width
        #cy = (gt_boxes[0, 2] + gt_boxes[0, 0]) / 2.0 * init_img.height
        startx = gt_boxes[0, 1] * init_img_width - pad_x
        starty = gt_boxes[0, 0] * init_img_height - pad_y
        endx = gt_boxes[0, 3] * init_img_width + pad_x
        endy = gt_boxes[0, 2] * init_img_height + pad_y
        ## so here: just slightly enlarge the bbox cropping area

        left_pad = max(0, int(-startx))
        top_pad = max(0, int(-starty))
        right_pad = max(0, int(endx - init_img_width + 1))
        bottom_pad = max(0, int(endy - init_img_height + 1)) ## prevent bbox out of init_img

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

        img1_xiaobai = Image.fromarray(img1_xiaobai)
        im = np.array(init_img) ## backup for crop pos/neg sample region for training

        # gt_boxes resize
        init_img_crop = img1_xiaobai.crop(np.int32([startx, starty, endx, endy]))
        init_img_crop = init_img_crop.resize([128,128], resample=Image.BILINEAR)
        self.last_gt = init_gt  # ymin xmin ymax xmax

        self.init_img_array = np.array(init_img_crop)
        self.init_feature_maps = self.sess.run(self.initFeatOp, feed_dict={self.initInputOp:self.init_img_array}) ## input the template
        ## self.init_feature_maps-> [1,1,1,512] for template
        ## here we get the init_feature_maps of template
        ## notice: just maintain as a filter !!!



        ##-------------------------------------------------------------------------------------------------


        self.target_w = init_gt[3] - init_gt[1]
        self.target_h = init_gt[2] - init_gt[0] ## record current w,h

        self.first_w = init_gt[3] - init_gt[1]
        self.first_h = init_gt[2] - init_gt[0]
        self.pos_regions_record = []
        self.neg_regions_record = []
        self.i = 0

        self.startx = 0
        self.starty = 0

    ##-------------------------------------------------------------------------------------------------

    def track(self, image):
        self.i += 1
        # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)  # seems that  RGB/BGR is not a problem
        cur_ori_img = Image.fromarray(image)
        # if self.expand_channel:
        #     cur_ori_img = np.array(cur_ori_img)
        #     cur_ori_img = np.expand_dims(cur_ori_img, axis=2)
        #     cur_ori_img = np.repeat(cur_ori_img, repeats=3, axis=2)
        #     cur_ori_img = Image.fromarray(cur_ori_img)
        cur_ori_img_array = np.array(cur_ori_img) ##(h,w,3)

        cropped_img, last_gt_norm, win_loc, scale = crop_search_region(cur_ori_img, self.last_gt, 300,
                                                                       crop_size=[self.target_h,self.target_w], mean_rgb=128)

        cur_img_array = np.array(cropped_img)
        detection_box_ori, scores = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                             feed_dict={self.input_cur_image: cur_img_array,
                                                        self.initConstantOp: self.init_feature_maps})
        # detection_box = detection_box[0]
        #print scores.shape ## the R scores
        #print scores[0]
        #print detection_box_ori
        #print '\n'

        ## notice: win_loc [y_min,x_min]  scale [height_scale, width_scale]


        detection_box_ori[:, 0] = detection_box_ori[:, 0] * scale[0] + win_loc[0]
        detection_box_ori[:, 1] = detection_box_ori[:, 1] * scale[1] + win_loc[1]
        detection_box_ori[:, 2] = detection_box_ori[:, 2] * scale[0] + win_loc[0]
        detection_box_ori[:, 3] = detection_box_ori[:, 3] * scale[1] + win_loc[1]
        ## look fn:crop_search_region
        ## just turn the bbox  in search region coordinate --> origin img(search region) coordinate
        clip_far = True

        if clip_far:
            rank = np.argsort(scores)
            k = 20
            candidates = rank[0, -k:]  ## top-20
            pixel_count = np.zeros((k,))
            for ii in range(k):
                bb = detection_box_ori[candidates[ii], :].copy()

                ## 0-y1   1-x1   2-y2   3-x2
                x1 = max(self.last_gt[1], bb[1])
                y1 = max(self.last_gt[0], bb[0])
                x2 = min(self.last_gt[3], bb[3])
                y2 = min(self.last_gt[2], bb[2])
                ## IOU ?
                pixel_count[ii] = (x2 - x1) * (y2 - y1) / float(
                    (self.last_gt[2] - self.last_gt[0]) * (self.last_gt[3] - self.last_gt[1]) +
                    (bb[3] - bb[1]) * (bb[2] - bb[0]) -
                    (x2 - x1) * (y2 - y1)
                )

            threshold = 0.4
            passed = pixel_count > (threshold)  ## throw out the too far proposals (w.r.t last gt)
            if np.sum(passed) > 0:
                candidates_left = candidates[passed]
                max_idx = candidates_left[np.argmax(scores[0, candidates_left])]
            else:
                max_idx = 0
        else:
            max_idx = 0

        clip_in_img = False
        if clip_in_img:
            search_box1 = detection_box_ori[max_idx]
            search_box1[0] = np.clip(search_box1[0], 0, cur_ori_img.height - 1)
            search_box1[2] = np.clip(search_box1[2], 0, cur_ori_img.height - 1)
            search_box1[1] = np.clip(search_box1[1], 0, cur_ori_img.width - 1)
            search_box1[3] = np.clip(search_box1[3], 0, cur_ori_img.width - 1)  ## y_min x_min y_max x_max

            # if (search_box1[0] == search_box1[2]) or (search_box1[1] == search_box1[3]): ## ???
            #     score_max = -1
            # else:
            #     search_box1 = [search_box1[1], search_box1[0], search_box1[3] - search_box1[1],
            #                    search_box1[2] - search_box1[0]] ## [x,y,w,h]
            #     search_box1 = np.reshape(search_box1, (1, 4))


            ## get highest box
            detection_box = np.reshape(search_box1, (4,))
        else:
            search_box1 = detection_box_ori[max_idx]
            detection_box = np.reshape(search_box1, (4,))


        score_threshold = False
        ## -------------------------
        if score_threshold:
            if scores[0, max_idx] < 0.3:
                x_c = (detection_box[3] + detection_box[1]) / 2.0
                y_c = (detection_box[0] + detection_box[2]) / 2.0
                w1 = self.last_gt[3] - self.last_gt[1]
                h1 = self.last_gt[2] - self.last_gt[0]
                x1 = x_c - w1 / 2.0
                y1 = y_c - h1 / 2.0
                x2 = x_c + w1 / 2.0
                y2 = y_c + h1 / 2.0
                self.last_gt = np.float32([y1, x1, y2, x2])
                ## keep the center, and use w,h of last_gt  (cannot understand)
            else:
                self.last_gt = detection_box
                self.target_w = detection_box[3] - detection_box[1]
                self.target_h = detection_box[2] - detection_box[0]

            ## y_min x_min y_max x_max
            if self.last_gt[0] < 0:
                self.last_gt[0] = 0
                self.last_gt[2] = self.target_h
            if self.last_gt[1] < 0:
                self.last_gt[1] = 0
                self.last_gt[3] = self.target_w
            if self.last_gt[2] > cur_ori_img.height:
                self.last_gt[2] = cur_ori_img.height - 1
                self.last_gt[0] = cur_ori_img.height - 1 - self.target_h
            if self.last_gt[3] > cur_ori_img.width:
                self.last_gt[3] = cur_ori_img.width - 1
                self.last_gt[1] = cur_ori_img.width - 1 - self.target_w

            self.target_w = (self.last_gt[3] - self.last_gt[1])
            self.target_h = (self.last_gt[2] - self.last_gt[0])
            width = self.last_gt[3] - self.last_gt[1]
            height = self.last_gt[2] - self.last_gt[0]
        else:
        ## -------------------------
            self.last_gt = detection_box
            # self.target_w = detection_box[3] - detection_box[1]
            # self.target_h = detection_box[2] - detection_box[0]
            width = self.last_gt[3] - self.last_gt[1]
            height = self.last_gt[2] - self.last_gt[0]


        ## show the detection result
        ## show_res(image, np.array(self.last_gt, dtype=np.int32), 'MBMD Tracker', score=scores[0,max_idx],score_max=0)



        return vot.Rectangle(float(self.last_gt[1]), float(self.last_gt[0]), float(width), float(height)), 0#scores[0,max_idx]

import time
import glob

#
# data_root = ""
# with open('otb_seq.txt') as f:
#     seq_list = f.readlines()
#     for i,seq in enumerate(seq_list):
#         seq_list[i] = seq.rstrip()
#
# print seq_list


with open('otb_seq.txt') as f:
    seq_list = f.readlines()
    for i,seq in enumerate(seq_list):
        seq_list[i] = seq.rstrip()

data_root = "../t_data/otb_data/"

n = -1

for idx,seq_name in enumerate(seq_list):
    print '\n {} {}'.format(idx,seq_name)

    if idx < n:
        continue

    tf.reset_default_graph()

    seq_path = os.path.join(data_root,seq_name)
    img_list = glob.glob(seq_path + '/img/*.jpg')
    img_list = sorted(img_list)

    region_file = os.path.join(seq_path, 'groundtruth_rect.txt')

    handle = vot.VOT("rectangle", img_list, region_file, seq_name)
    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = cv2.imread(imagefile)
    print len(handle)
    print image.shape
    print "init img: {}".format(imagefile)
    t1 = time.time()
    tracker = MobileTracker(image, selection)
    t2 = time.time()
    print "\ntracker initial finish, time cost {}s\n".format(t2 - t1)

    t3 = time.time()
    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        print (imagefile)
        image = cv2.imread(imagefile)
        region, confidence = tracker.track(image)
        handle.report(region, confidence)
    t4 = time.time()
    print "finish, time cost {}s per img".format((t4 - t3) / len(handle))
