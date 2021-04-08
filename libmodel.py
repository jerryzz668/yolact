from torchvision.transforms import Normalize, ToTensor
import numpy as np
import datetime
import torch
import cv2
import json
import logging
import math
from PIL import Image
from collections import namedtuple
from configparser import ConfigParser
import lib.libcom_pb2 as libcom__pb2
from models.experimental import attempt_load
from lib.libutils import  non_max_suppression, scale_coords, letterbox, hwc_to_chw, resize_and_crop, normalize
from torchvision import transforms
import os

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


class CombineModel(BaseModel):
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

        # detect model
        model1 = attempt_load(self.model_cfg['model_detect_path'])
        self.names = model1.module.names if hasattr(model1, 'module') else model1.names
        model1 = model1.cuda()
        model1 = model1.eval()
        model1.half()
        self.model_detect = model1

        print("detect_" + "model_" + str(self.model_id) + " is started")

        # seg model
        if self.model_out_with_mask==True and self.seg_model_enable == True:
            try:
                model2 = torch.load(self.model_cfg['model_seg_path'])
                model2 = model2.cuda()
                model2 = model2.eval()
                self.seg_model = model2
                print("segment_" + "model_" + str(self.model_id) + " is started")
            except:
                self.seg_model_enable = False
                logging.error("model_" + str(self.model_id) + "'s segment model is not exist")

        # class model
        if self.class_model_enable == True:
            try:
                model3 = torch.load(self.model_cfg['model_class_path'])
                model3 = model3.cuda()
                model3 = model3.eval()
                self.classify_model = model3
                print("class_" + "model_" + str(self.model_id) + " is started")
            except:
                self.class_model_enable = False
                logging.error("model_" + str(self.model_id) + "'s class model is not exist")

        print("all_" + "model_" + str(self.model_id) + " is started")

    def _trans_list2points(self, point_list, x_offset, y_offset):
        """
        transform a point list to proto Point list
        """
        return list(map(lambda p:libcom__pb2.Point(x=p[0]+x_offset,y=p[1]+y_offset),point_list))

    def imgCheck(self, image, max_stride, pre_treat_mode=True):
        max_stride = int(max_stride)
        if pre_treat_mode:
            new_img = image[:math.floor(image.shape[:-1][0] / max_stride) * max_stride,
                            :math.floor(image.shape[:-1][1] / max_stride) * max_stride,
                            :]

            s_h = image.shape[0]
            s_w = image.shape[1]

            t_h = new_img.shape[0]
            t_w = new_img.shape[1]

            scale_h = t_h * 1.0 / s_h
            scale_w = t_w * 1.0 / s_w
        else:
            new_img = letterbox(image, (math.ceil(image.shape[:-1][0] / max_stride) * max_stride,
                                             math.ceil(image.shape[:-1][0] / max_stride) * max_stride))[0]
            scale_h = 1.0
            scale_w = 1.0

        return new_img, scale_h, scale_w

    def detect(self, image):
        model=self.model_detect
        new_img, scale_h, scale_w = self.imgCheck(image, self.model_detect.stride.max(), self.pre_treat_mode)
        # Convert
        img = new_img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).cuda()
        img = img.half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img, augment=False)[0]
        # Apply NMS
        pred = non_max_suppression(pred, float(self.conf_threshold), float(self.iou_threshold), classes=None, agnostic=False)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], new_img.shape).round()
        return det, scale_h, scale_w

    def seg_detect(self, image, scale=1, x_offset=0, y_offset=0):

        model = self.seg_model
        img = Image.fromarray(image)
        img = resize_and_crop(img, scale)
        img = ToTensor()(img)
        Normalize([.485, .456, .406], [.229, .224, .225])(img)
        img = img.unsqueeze(0)
        img = img.cuda()
        output = model(img)
        sem_mask = output.squeeze().detach()
        mask_img = self.sem_mask_acc(sem_mask, scale, image.shape)
        return mask_img

    def classify_detect(self, image0, x_center, y_center, crop_width=224):
        left_x = int(x_center - crop_width / 2)
        left_y = int(y_center - crop_width / 2)
        if left_x < 0:
            left_x = 0
        else:
            left_x = left_x
        if left_x + crop_width > image0.shape[1]:
            left_x = image0.shape[1] - crop_width
        if left_y < 0:
            left_y = 0
        else:
            left_y = left_y
        if left_y + crop_width > image0.shape[0]:
            left_y = image0.shape[0] - crop_width

        image = image0[left_y:left_y + crop_width,
                       left_x:left_x + crop_width]

        image3 = image
        if len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_model = self.class_model

        print("image:", image.shape)
        # print("full_img:", full_img.shape)
        full_img = Image.fromarray(image)
        print("full_img:", full_img.size)
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        input = transform_train(full_img)
        print("input:::::::::", input.shape)
        input = input.unsqueeze(0)
        # print("input:", input.shape)
        input = input.cuda()
        input = torch.autograd.Variable(input)
        output = class_model(input)
        #logit = F.softmax(output, dim=1)
        a = output.data
        _, pre = a.topk(1, dim=1, largest=True)
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S_%f')
        image_NG_path = "./crop/NG"
        image_OK_path = "./crop/OK"

        image_NG_name = image_NG_path + "/NG_" + time_str + ".jpg"
        image_OK_name = image_OK_path + "/OK_" + time_str + ".jpg"

        if os.path.exists(image_NG_path) == False:
            os.makedirs(image_NG_path)
        # make new output folder
        if os.path.exists(image_OK_path) == False:
            os.makedirs(image_OK_path)  # make new output folder

        if pre == 0:
            pre = 0
            cv2.imwrite(image_NG_name, image3)
        elif pre == 1:
            pre = 1
            cv2.imwrite(image_OK_name, image3)
        return pre

    def sem_mask_acc(self, sem_mask, scale, img_shape, mask_ratio=0.2, acc_enable=False):

        if acc_enable:
            points = sem_mask > mask_ratio
            points = torch.nonzero(points)
            image = np.zeros((img_shape[0], img_shape[1]), dtype=np.uint8)
            if len(points)>0:
                try:
                    points = points.cpu().numpy()
                    points[:, [0, 1]] = points[:, [1, 0]]
                    points = points.reshape(-1, 1, 2)
                    cv2.drawContours(image, points, -1, 255, 3)
                    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
                    image=cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
                except:
                    logging.debug("seg error")
                    print("seg error")

            image = cv2.resize(image, (0, 0), fx=1/scale, fy=1/scale)
            ret, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
        else:
            mask_np = sem_mask.cpu().numpy()
            mask_np = cv2.resize(mask_np, (0, 0), fx=1/scale, fy=1/scale)
            mask_np = mask_np > 0.5
            image   = mask_np.astype(np.bool).astype(np.uint8) * 255
        return image

    def inference(self, image_np, photo_id, x_offset=0, y_offset=0):
        # input
        if len(image_np.shape)==2:
            image_np=cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        # detect
        pre, scale_h, scale_w = self.detect(image_np)
        # result
        self.result = self._merge_result(image_np, pre, scale_h, scale_w, x_offset, y_offset)

    def _merge_result(self, image_np, pre, scale_h, scale_w, x_offset, y_offset):
        total_boxes, total_labels, total_masks, new_points = [], [], [], []
        if pre != None:
            for *xyxy, conf, cls in reversed(pre):

                top_x    = int(xyxy[0])# * 1.0 * scale_w * 0.995
                bottom_x = int(xyxy[2])# * 1.0* scale_w * 0.995

                top_y    = int(xyxy[1])# * 1.0  * scale_h * 1
                bottom_y = int(xyxy[3])# * 1.0  * scale_h * 1

                x_center = (bottom_x + top_x) // 2
                y_center = (bottom_y + top_y) // 2

                if self.class_model_enable:

                    if self.names[int(cls)] in self.class_label_name:
                        print("self.names[int(cls)]::::", self.names[int(cls)])
                        flag = self.classify_detect(image_np, x_center, y_center, crop_width=224)
                        if flag == 1:
                            continue

                total_boxes.append(
                    [top_x + x_offset, top_y + y_offset, bottom_x + x_offset, bottom_y + y_offset, float(conf)])

                total_labels.append(int(cls))

                mask_img, points = self._generate_mask(image_np, top_x, top_y, bottom_x, bottom_y, x_offset,
                                                      y_offset, self.model_out_with_mask, self.seg_model_enable, self.box_add_length_to_mask)
                total_masks.append(mask_img)
                new_points.append(points)

        Result = namedtuple('Result', ['masks', 'bboxes', 'labels', 'points'])
        result = Result(masks=total_masks, bboxes=total_boxes, labels=total_labels, points=new_points)
        torch.cuda.empty_cache()
        return result

    def _generate_mask(self, image_np, top_x, top_y, bottom_x, bottom_y,
                      x_offset, y_offset, model_out_with_mask, seg_model_enable, box_add_length_to_mask):
        if model_out_with_mask == False:
            mask_img = []
            points = []
        else:
            width = bottom_x - top_x
            height = bottom_y - top_y
            segimg, left_x, left_y = self.make_segimg(image_np, top_x, top_y, bottom_x, bottom_y, box_add_length_to_mask)

            if seg_model_enable == True:
                result_img = self.seg_detect(segimg)
                mask_img = result_img[top_y - left_y:top_y - left_y + height, top_x - left_x:top_x - left_x + width]

            else:# 采用box 作为 mask
                result_img = segimg[:, :, 0]
                out_img = result_img[top_y - left_y:top_y - left_y + height, top_x - left_x:top_x - left_x + width]
                mask_img = out_img.astype(np.bool).astype(np.uint8) * 0

            if np.sum(mask_img[:, :] > 0) == 0:
                mask_img = mask_img + 255
                mask_img[0, :] = 0
                mask_img[-1, :] = 0
                mask_img[:, 0] = 0
                mask_img[:, -1] = 0
            contours, _ = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            points = self._trans_list2points(contours[0].reshape((-1, 2)), top_x + x_offset, top_y + y_offset)
        return mask_img, points

    def make_segimg(self, image_np, top_x, top_y, bottom_x, bottom_y, box_add_length_to_mask):

        crop_width = bottom_x - top_x + box_add_length_to_mask * 2
        crop_height = bottom_y - top_y + box_add_length_to_mask * 2

        if top_x - box_add_length_to_mask < 0:
            left_x = 0
        else:
            left_x = top_x - box_add_length_to_mask

        if left_x + crop_width > image_np.shape[1]:
            left_x = image_np.shape[1] - crop_width

        if top_y - box_add_length_to_mask < 0:
            left_y = 0
        else:
            left_y = top_y - box_add_length_to_mask

        if left_y + crop_height > image_np.shape[0]:
            left_y = image_np.shape[0] - crop_height

        segimg = image_np[left_y: left_y + crop_height,
                 left_x:  left_x + crop_width,
                 :]

        return segimg, left_x, left_y

