# -- coding: utf-8 --

import cv2
import os
import sys
import tensorflow as tf
sys.path.append('./lib')
sys.path.append('./lib/slim')

from lib.object_detection.protos import pipeline_pb2
from lib.object_detection.core import box_list
from lib.object_detection.core import box_list_ops
from lib.object_detection.builders import preprocessor_builder, optimizer_builder
from lib.object_detection.utils import variables_helper
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
import vot
from vggm import vggM
from core.model_builder import build_man_model
from sample_generator import *
from google.protobuf import text_format
from train.data_utils import draw_box,draw_mulitbox,iou_y1x1y2x2
os.environ["CUDA_VISIBLE_DEVICES"]="1"





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

    # input_init_gt_box = tf.constant(np.zeros((batchsize, 1, 4)), dtype=tf.float32)
    input_init_gt_box = tf.placeholder(dtype=tf.float32, shape=[batchsize, 1, 4])

    init_gt_box = tf.reshape(input_init_gt_box, shape=[batchsize,1,4])  ## (B,1,4)
    groundtruth_classes = tf.placeholder(dtype=tf.int32, shape=[batchsize, 1, 1]) ## (B,1,1)

    model.provide_groundtruth(init_gt_box,
                              groundtruth_classes,
                              None)

    prediction_dict = model.predict(preprocessed_init_image, preprocessed_images, istraining=False)


    detections = model.postprocess(prediction_dict)  ## NMS

    return prediction_dict ,detections['detection_boxes'] * 300, detections['detection_scores'], \
           input_cur_image,input_init_image,input_init_gt_box,groundtruth_classes


class MobileTracker(object):
    def __init__(self,batchsize=2,train_phase=True):
        training = train_phase
        self.bs = batchsize
        self.i = 0
        self.global_step = tf.train.get_or_create_global_step()

        config_file = 'MBMD_vot_model/model/ssd_mobilenet_video.config'
        checkpoint_dir = '/home/yuzhe/PycharmProjects/test_mbmd/MBMD_vot_model/model/dump'

        model_config, train_config, input_config, eval_config = get_configs_from_pipeline_file(config_file)
        model = build_man_model(model_config=model_config, is_training=False)

        ##-------------------------------------------------------------------------------------------------


        prediction_dict, self.pre_box_tensor, self.scores_tensor, \
        self.input_cur_image, self.initInputOp, self.cur_gt_op,self.cur_label_op= \
                                                build_box_predictor(model,batchsize=self.bs)

        self.losses_dict = model.loss(prediction_dict)
        total_loss = self.losses_dict['localization_loss'] + self.losses_dict['classification_loss']

        training_optimizer = optimizer_builder.build(train_config.optimizer, set())
        self.train_op = training_optimizer.minimize(total_loss, global_step=self.global_step)



        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        variables_to_restore = tf.global_variables()
        self.var_all = variables_to_restore

        self.log_f = open('log_vid.txt','w+')

        pretrained = True
        if pretrained:
            var_map = model.restore_map(
                from_detection_checkpoint=train_config.from_detection_checkpoint)
            var_map_init = model.restore_init_map(
                from_detection_checkpoint=train_config.from_detection_checkpoint)

            available_var_map = (variables_helper.
                get_variables_available_in_checkpoint(
                var_map, train_config.fine_tune_checkpoint))
            available_var_map_init = (variables_helper.
                get_variables_available_in_checkpoint(
                var_map_init, train_config.fine_tune_checkpoint))

            feat_extract_saver = tf.train.Saver(available_var_map)
            init_saver = tf.train.Saver(available_var_map_init)

            feat_extract_saver.restore(self.sess, train_config.fine_tune_checkpoint)
            init_saver.restore(self.sess, train_config.fine_tune_checkpoint)





    ##-------------------------------------------------------------------------------------------------

    def test_batch_input(self, init_img_array, cur_img_array, cur_gt,cur_label):
        """

        : init_img_array: (B,128,128,3)
        : cur_img_array: (B,300,300,3)
        : cur_gt: (B,1,4)
        """


        self.i += 1
        cur_gt = cur_gt  # (B,1,4)

        detection_box_ori, scores = self.sess.run([self.pre_box_tensor, self.scores_tensor],
                                                  feed_dict={self.input_cur_image: cur_img_array,
                                                             self.initInputOp: init_img_array,
                                                             self.cur_gt_op:cur_gt,
                                                             self.cur_label_op:cur_label})

        # print detection_box_ori.shape
        # print scores.shape
        # (2, 100, 4)
        # (2, 100)

        sample1_img = cur_img_array[0]
        sample1_det = detection_box_ori[0][0]
        sample1_gt = cur_gt[0][0] * 300.0


        cur_iou = iou_y1x1y2x2(sample1_det,sample1_gt)

        print 'iou {}'.format(cur_iou)

        box_show = np.array([sample1_det[1], sample1_det[0], sample1_det[3], sample1_det[2]])
        gt_show = np.array([sample1_gt[1],sample1_gt[0],sample1_gt[3],sample1_gt[2]])
        cur_ori_img = Image.fromarray(sample1_img[..., ::-1])
        draw_mulitbox(cur_ori_img, [box_show.astype(np.int32),gt_show.astype(np.int32)], img_path='tmp3/{}.jpg'.format(self.i))


    def train_step(self,init_img_array, cur_img_array, cur_gt,cur_label):

        detection_box_ori, loss_dict, _ = self.sess.run([self.pre_box_tensor, self.losses_dict, self.train_op],
                                                  feed_dict={self.input_cur_image: cur_img_array,
                                                             self.initInputOp: init_img_array,
                                                             self.cur_gt_op:cur_gt,
                                                             self.cur_label_op:cur_label})
        print loss_dict

        sample1_gt = cur_gt[0][0]*300
        sample1_det = detection_box_ori[0][0]
        # print sample1_det
        cur_iou = iou_y1x1y2x2(sample1_det, sample1_gt)

        print 'iou {}'.format(cur_iou)
        
        total_loss = loss_dict['localization_loss'] + loss_dict['classification_loss']
        self.i += 1
        if self.i % 200 == 0:
            print '\n\nwrite in log file\n\n'
            with open('log_vid.txt','a') as f:
                f.write('iter {} loss={}  iou={}\n'.format(self.i,total_loss,cur_iou))


if __name__ == '__main__':

    import torch
    #from train.data_loader import basketball_dataset,otb_collate
    from pair_dataset import vid_dataset,vid_collate

    bs = 16

    tracker = MobileTracker(bs,train_phase=True)

    test_loader = vid_dataset('data_prepare/vid_all.pkl','../t_data/vid_data/Data/VID/train/')
    print (len(test_loader))
    data_loader = torch.utils.data.DataLoader(test_loader,batch_size=bs, shuffle=True,num_workers=1,
                                              collate_fn=vid_collate,drop_last=True)

    model_saver = tf.train.Saver(max_to_keep=2,var_list=tracker.var_all)

    from datetime import datetime

    for i in range(2):
        with open('log_vid.txt','a') as f:
            time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            f.write('\nepoch begin {}\n'.format(time_string))
        for idx, (templates, imgs, gts, labels) in enumerate(data_loader):
            print "in Epoch {} iter {}".format(i,idx)
            # print(templates.shape)
            # tracker.test_batch_input(templates, imgs, gts)

            # sample1_img = imgs[0]
            # sample1_tp = templates[0]
            # sample1_gt = gts[0][0]*300
            # gt_show = np.array([sample1_gt[1], sample1_gt[0], sample1_gt[3], sample1_gt[2]])
            # cur_ori_img = Image.fromarray(sample1_img[..., ::-1])
            # draw_mulitbox(cur_ori_img, [gt_show.astype(np.int32)], img_path='tmp3/{}.jpg'.format(idx))
            # cv2.imwrite('tmp3/{}_tp.jpg'.format(idx),sample1_tp)

            tracker.train_step(templates, imgs, gts,labels)
            if idx % 5000 == 0:
                save_path = model_saver.save(tracker.sess, 'model/vid_model/model.ckpt')
                print 'save model in {}'.format(save_path)
            if idx == 500000:
                exit()






