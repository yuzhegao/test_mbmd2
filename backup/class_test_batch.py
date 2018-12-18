# -- coding: utf-8 --

import cv2
import os
import sys
import tensorflow as tf
sys.path.append('../')
sys.path.append('../lib')
sys.path.append('../lib/slim')

from lib.object_detection.protos import pipeline_pb2
from lib.object_detection.core import box_list
from lib.object_detection.core import box_list_ops
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import vot
from vggm import vggM
from core.model_builder import build_man_model
from sample_generator import *
from google.protobuf import text_format
from train.data_utils import draw_box,draw_mulitbox,iou_y1x1y2x2
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


def restore_model(sess, checkpoint_path, variables_to_restore):
    # variables_to_restore = tf.global_variables()
    name_to_var_dict = dict([(var.op.name, var) for var in variables_to_restore
                             if not var.op.name.endswith('Momentum')])
    saver = tf.train.Saver(name_to_var_dict)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    saver.restore(sess, latest_checkpoint)

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
    ## now　resize and get "resize_scale_rate"

    # win = sp.misc.imresize(unscaled_win, [win_size, win_size])
    return win, np.array([gt_y_min, gt_x_min, gt_y_max, gt_x_max]), win_loc, [height_scale, width_scale]
    # return win, np.array([gt_x_min, gt_y_min, gt_x_max, gt_y_max]), diag, np.array(win_loc)


def build_box_predictor(model,batchsize=1):

    input_init_image = tf.placeholder(dtype=tf.uint8, shape=[batchsize,128, 128, 3])  ## template patch (B,128,128,3)
    float_init_image = tf.to_float(input_init_image)
    float_init_image = tf.expand_dims(float_init_image, axis=0) # (B,1,128,128,3)
    preprocessed_init_image = model.preprocess(float_init_image, [128, 128])   # (B,1,128,128,3)
    ## resize + mobilenet.preprocess

    input_cur_image = tf.placeholder(dtype=tf.uint8, shape=[batchsize,300, 300, 3])
    float_images = tf.to_float(input_cur_image)
    preprocessed_images = model.preprocess(float_images) # (B,300,300,3)
    preprocessed_images = tf.expand_dims(preprocessed_images, axis=1) # (B,1,300,300,3)

    input_init_gt_box = tf.constant(np.zeros((batchsize,1, 4)), dtype=tf.float32)
    init_gt_box = tf.reshape(input_init_gt_box, shape=[batchsize,1,4])  ## (B,1,4)
    groundtruth_classes = tf.ones(dtype=tf.float32, shape=[batchsize, 1, 1]) ## (B,1,1)

    model.provide_groundtruth(init_gt_box,
                              groundtruth_classes,
                              None)

    prediction_dict = model.predict(preprocessed_init_image, preprocessed_images, istraining=False)


    detections = model.postprocess(prediction_dict)  ## NMS

    return detections['detection_boxes'] * 300, detections['detection_scores'], input_cur_image,input_init_image


class MobileTracker(object):
    def __init__(self,batchsize=2):
        init_training = True
        self.bs = batchsize
        self.i = 0

        config_file = '/home/yuzhe/PycharmProjects/test_mbmd2/MBMD_vot_model/model/ssd_mobilenet_video.config'
        checkpoint_dir = '/home/yuzhe/PycharmProjects/test_mbmd2/model/ssd_mobilenet_video1/'
        # checkpoint_dir = '/home/yuzhe/PycharmProjects/test_mbmd/MBMD_vot_model/model/dump'

        model_config, train_config, input_config, eval_config = get_configs_from_pipeline_file(config_file)
        model = build_man_model(model_config=model_config, is_training=False)

        ##-------------------------------------------------------------------------------------------------


        self.pre_box_tensor, self.scores_tensor, self.input_cur_image, self.initInputOp = build_box_predictor(model,batchsize=self.bs)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        variables_to_restore = tf.global_variables()


        restore_model(self.sess, checkpoint_dir, variables_to_restore) ## restore the siamese network checkpoint



    ##-------------------------------------------------------------------------------------------------
    def test(self,init_img_array,init_gt,image,last_gt,cur_gt):
        """
        init_gt: x1 y1 x2 y2 *300
        gt_cur: x1 y1 x2 y2 *300
        """

        self.i += 1

        # --------------------------------------------------------------------------------------------------------
        # process init img to template

        init_img = Image.fromarray(init_img_array)  ## the first frame
        init_img_array = np.array(init_img)
        if init_img_array.ndim < 3:
            init_img_array = np.expand_dims(init_img_array, axis=2)
            init_img_array = np.repeat(init_img_array, repeats=3, axis=2)
            init_img = Image.fromarray(init_img_array)

        ## init_gt  # ymin xmin ymax xmax

        ## if not padding, the size of predict box will not be very correct
        Padding = True

        img1_xiaobai = np.array(init_img)
        gt_boxes = np.zeros((1, 4))
        gt_boxes[0, 0] = init_gt[0] / float(init_img.height)
        gt_boxes[0, 1] = init_gt[1] / float(init_img.width)
        gt_boxes[0, 2] = init_gt[2] / float(init_img.height)
        gt_boxes[0, 3] = init_gt[3] / float(init_img.width)

        if Padding:
            pad_x = 36.0 / 264.0 * (gt_boxes[0, 3] - gt_boxes[0, 1]) * init_img.width
            pad_y = 36.0 / 264.0 * (gt_boxes[0, 2] - gt_boxes[0, 0]) * init_img.height
            startx = gt_boxes[0, 1] * init_img.width - pad_x
            starty = gt_boxes[0, 0] * init_img.height - pad_y
            endx = gt_boxes[0, 3] * init_img.width + pad_x
            endy = gt_boxes[0, 2] * init_img.height + pad_y

            left_pad = max(0, int(-startx))
            top_pad = max(0, int(-starty))
            right_pad = max(0, int(endx - init_img.width + 1))
            bottom_pad = max(0, int(endy - init_img.height + 1))  ## prevent bbox out of init_img

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
            startx = gt_boxes[0, 1] * init_img.width
            starty = gt_boxes[0, 0] * init_img.height
            endx = gt_boxes[0, 3] * init_img.width
            endy = gt_boxes[0, 2] * init_img.height

        img1_xiaobai = Image.fromarray(img1_xiaobai)

        # gt_boxes resize
        init_img_crop = img1_xiaobai.crop(np.int32([startx, starty, endx, endy]))
        init_img_crop = init_img_crop.resize([128, 128], resample=Image.BILINEAR)

        init_img_array = np.array(init_img_crop)

        ## --------------------------------------------------------------------------------------------------------


        cur_ori_img = Image.fromarray(image)
        if image.ndim < 3:
            cur_ori_img = np.array(cur_ori_img)
            cur_ori_img = np.expand_dims(cur_ori_img, axis=2)
            cur_ori_img = np.repeat(cur_ori_img, repeats=3, axis=2)
            cur_ori_img = Image.fromarray(cur_ori_img)

        cropped_img, last_gt_norm, win_loc, scale = crop_search_region(cur_ori_img, last_gt, 300, mean_rgb=128)

        cur_img_array = np.array(cropped_img)
        detection_box_ori, scores = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                  feed_dict={self.input_cur_image: cur_img_array,
                                                             self.initInputOp: init_img_array})


        detection_box_ori[:, 0] = detection_box_ori[:, 0] * scale[0] + win_loc[0]
        detection_box_ori[:, 1] = detection_box_ori[:, 1] * scale[1] + win_loc[1]
        detection_box_ori[:, 2] = detection_box_ori[:, 2] * scale[0] + win_loc[0]
        detection_box_ori[:, 3] = detection_box_ori[:, 3] * scale[1] + win_loc[1]

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

        search_box1 = detection_box_ori[max_idx]
        detection_box = np.reshape(search_box1, (4,))


        self.last_gt = detection_box
        self.target_w = detection_box[3] - detection_box[1]
        self.target_h = detection_box[2] - detection_box[0]
        width = self.last_gt[3] - self.last_gt[1]
        height = self.last_gt[2] - self.last_gt[0]


        cur_iou = iou_y1x1y2x2(cur_gt,self.last_gt)

        print 'iou {}'.format(cur_iou)

        ## show the detection result
        box_show = np.array([self.last_gt[1], self.last_gt[0], self.last_gt[3], self.last_gt[2]])
        gt_show = np.array([cur_gt[1],cur_gt[0],cur_gt[3],cur_gt[2]])
        cur_ori_img = Image.fromarray(image[..., ::-1])
        draw_mulitbox(cur_ori_img, [box_show.astype(np.int32),gt_show.astype(np.int32)], img_path='tmp3/{}.jpg'.format(self.i))
        cv2.imshow('1', cv2.imread('tmp3/{}.jpg'.format(self.i)))
        cv2.waitKey(1)

        return vot.Rectangle(float(self.last_gt[1]), float(self.last_gt[0]), float(width),
                             float(height)), 0  # scores[0,max_idx]



    def test_batch_input(self, init_img_array, cur_img_array, cur_gt):

        self.i += 1
        cur_gt = cur_gt * 300.0 # (B,1,4)

        detection_box_ori, scores = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                  feed_dict={self.input_cur_image: cur_img_array,
                                                             self.initInputOp: init_img_array})

        # print detection_box_ori.shape
        # print scores.shape
        # (2, 100, 4)
        # (2, 100)

        sample_id = 0

        sample1_img = cur_img_array[sample_id]
        sample1_det = detection_box_ori[sample_id][0]
        sample1_gt = cur_gt[sample_id][0]


        cur_iou = iou_y1x1y2x2(sample1_det,sample1_gt)

        print 'iou {}'.format(cur_iou)

        ## show the detection result
        box_show = np.array([sample1_det[1], sample1_det[0], sample1_det[3], sample1_det[2]])
        gt_show = np.array([sample1_gt[1],sample1_gt[0],sample1_gt[3],sample1_gt[2]])
        cur_ori_img = Image.fromarray(sample1_img[..., ::-1])
        draw_mulitbox(cur_ori_img, [box_show.astype(np.int32),gt_show.astype(np.int32)], img_path='../tmp3/{}.jpg'.format(self.i))

        # sample1_tp = init_img_array[sample_id]
        # cv2.imwrite('../tmp3/{}_tp.jpg'.format(self.i),sample1_tp)
        #
        # if self.i >=50:
        #     exit()

        return  0




if __name__ == '__main__':

    # handle = vot.VOT("rectangle")
    # init_region = handle.region()
    # imagefile = handle.frame()
    #
    # init_image = cv2.imread(imagefile)
    # print len(handle)
    # print init_image.shape
    #
    # init_gt1 = [init_region.x, init_region.y, init_region.width, init_region.height]
    # init_gt = [init_gt1[1], init_gt1[0], init_gt1[1] + init_gt1[3], init_gt1[0] + init_gt1[2]]  # ymin xmin ymax xmax
    #
    # tracker = MobileTracker(init_gt)
    #
    #
    # while True:
    #     imagefile = handle.frame()
    #     if not imagefile:
    #         break
    #
    #     cur_gt = handle.gt()
    #     cur_gt = np.array(cur_gt).astype(np.int32)
    #     cur_gt = np.array([cur_gt[1], cur_gt[0], cur_gt[1] + cur_gt[3], cur_gt[0] + cur_gt[2]])  ## x1 y1 x2 y2
    #
    #     print (imagefile)
    #     image = cv2.imread(imagefile)
    #     # image = np.array(Image.open(imagefile))   ## Notice: the cv2 and PIL difference !!!!
    #     region, confidence = tracker.test_input(init_image, init_gt, image, tracker.last_gt, cur_gt)
    #     handle.report(region, confidence)
    import torch
    from train.data_loader import basketball_dataset,otb_collate

    bs = 2

    tracker = MobileTracker(bs)

    test_loader = basketball_dataset('/home/yuzhe/Downloads/part_vot_seq/')
    print (len(test_loader))
    data_loader = torch.utils.data.DataLoader(test_loader,
                                              batch_size=bs, shuffle=False, num_workers=1, collate_fn=otb_collate)


    for idx, (templates, imgs, gts, labels) in enumerate(data_loader):
        print idx
        # print(templates.shape)
        tracker.test_batch_input(templates, imgs, gts)

        # sample1_img = imgs[0]
        # sample1_tp = templates[0]
        # sample1_gt = gts[0][0]*300
        # gt_show = np.array([sample1_gt[1], sample1_gt[0], sample1_gt[3], sample1_gt[2]])
        # cur_ori_img = Image.fromarray(sample1_img[..., ::-1])
        # draw_mulitbox(cur_ori_img, [gt_show.astype(np.int32)], img_path='tmp3/{}.jpg'.format(idx))
        # cv2.imwrite('tmp3/{}_tp.jpg'.format(idx),sample1_tp)






