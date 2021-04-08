# @Description:
# @Author     : zhangyan
# @Time       : 2021/3/15 4:23 PM

import torch
import cv2
import json
import logging
from configparser import ConfigParser

from yolact import Yolact

from data import cfg, set_cfg, set_dataset
from utils import timer
from layers.output_utils import postprocess, undo_image_transformation
from utils.augmentations import BaseTransform, FastBaseTransform, Resize
import numpy as np
from collections import namedtuple

class BaseModel(object):

    def __init__(self, model_id, cfg_path):
        # model id
        self.model_id = model_id
        self.cfg_path = cfg_path
        self.cfg_dic = self._config_parser()
        # basic config
        self.config_basic = self.cfg_dic['config_basic']
        # model_config_cycle_per_num
        self.model_config_cycle_per_num = int(self.config_basic['model_config_cycle_per_num'])
        # model all use one config
        if self.config_basic['model_all_enable'] == 'True':
            self.model_name = 'model_all'
            print(self.model_name + "_" + str(self.model_id))
        else:
            model_id_new = int(self.model_id) % self.model_config_cycle_per_num
            if model_id_new == 0: model_id_new=self.model_config_cycle_per_num
            self.model_name = 'model_{}'.format(str(model_id_new))
            print(self.model_name + '_' + str(self.model_id))
        # image pre treat mode
        if self.config_basic['pre_treat_cut'] == 'True':
            self.pre_treat_mode = True
        else:
            self.pre_treat_mode = False
        # model out with mask
        if self.config_basic['model_out_with_mask'] == 'True':
            self.model_out_with_mask = True
        else:
            self.model_out_with_mask = False
        # class model enable
        if self.config_basic['seg_model_enable'] == 'True':
            self.seg_model_enable = True
        else:
            self.seg_model_enable = False
        # box add length to as seg model input
        self.box_add_length_to_mask = int(self.config_basic['box_add_length_to_mask'])
        # class model enable
        if self.config_basic['class_model_enable'] == 'True':
            self.class_model_enable = True
        else:
            self.class_model_enable = False
        # model config
        self.model_cfg  = self.cfg_dic[self.model_name]
        self.conf_threshold = self.model_cfg['conf_threshold']
        self.iou_threshold  = self.model_cfg['iou_threshold']

        self.class_mapper = json.loads(self.model_cfg['class_mapper'])
        self.thresh_values = [float(x) for x in self.model_cfg['thresh_values'].split(',')]
        self.num_classes = int(self.model_cfg['num_classes'])

        if self.class_model_enable == True:
            try:
                self.class_label_name = [str(x) for x in self.model_cfg['class_label_name'].split(',')]
            except:
                self.class_model_enable = False
                logging.error("model_" + str(self.model_id) + "'s class_label_name is not exist")

        self.result = None

    def _config_parser(self):
        logging.info('config file loaded: {}'.format(self.cfg_path))
        conf = ConfigParser()
        conf.read(self.cfg_path)
        cfg_dict = {}
        for section in conf.sections():
            section_items = conf.items(section)
            section_dict = {}
            for pair in section_items:
                section_dict[pair[0]] = pair[1]
            cfg_dict[section] = section_dict

        return cfg_dict


class Yolact_Model(BaseModel):
    """
    base
    """
    def __init__(self, model_id, cfg_path):
        """初始化"""
        super().__init__(model_id, cfg_path)
        # 启动模型
        self.start_model()

    def start_model(self):
        """start model"""

        print('Loading model...', end='')
        net = Yolact()
        net.load_weights(self.model_cfg['model_detect_path'])
        net.eval()
        print(' Done.')
        net = net.cuda()
        self.model_yo = net

    def _trans_list2points(self, point_list, x_offset, y_offset):
        """
        transform a point list to proto Point list
        """
        return list(map(lambda p:libcom__pb2.Point(x=p[0]+x_offset,y=p[1]+y_offset),point_list))

    def modify_masks_boxes_classes(self, masks, boxes, classes, x_offset, y_offset):
        total_boxes, total_labels, total_masks, new_points = [], [], [], []
        num = boxes.shape[0]
        for i in range(num):
            masks_np = masks[i,:,:]
            masks_np = torch.squeeze(masks_np)
            mask_np = masks_np > 0.5
            mask_image = mask_np.astype(np.bool).astype(np.uint8) * 255

            total_boxes.append(boxes[i])

            total_labels.append(int(classes))

            contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = self._trans_list2points(contours[0].reshape((-1, 2)), boxes[i][0] + x_offset, boxes[i][1] + y_offset)
            new_points.append(points)

        Result = namedtuple('Result', ['masks', 'bboxes', 'labels', 'points'])
        result = Result(masks=total_masks, bboxes=total_boxes, labels=total_labels, points=new_points)
        torch.cuda.empty_cache()
        return result

    def detect(self, image):
        model=self.model_yo
        model.detect.fast_nms = True
        model.detect.use_cross_class_nms = False
        cfg.mask_proto_debug = False

        frame = torch.from_numpy(cv2.imread(image)).cuda().float()
        batch = FastBaseTransform()(frame.unsqueeze(0))
        preds = model(batch)

        img_gpu = frame / 255.0
        h, w, _ = frame.shape

        with timer.env('Postprocess'):
            save = cfg.rescore_bbox
            cfg.rescore_bbox = True
            t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=0)
            cfg.rescore_bbox = save

        with timer.env('Copy'):
            idx = t[1].argsort(0, descending=True)[:5]

            if cfg.eval_mask_branch:
                # Masks are drawn on the GPU, so don't copy
                masks = t[3][idx]
            classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]

        result = self.modify_masks_boxes_classes(masks, boxes, classes)

        return result









