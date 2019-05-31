import os
import json
import yaml
import numpy as np
from collections import defaultdict
from collections import Counter

from utils.label_map_tools import load_label_map
from utils.label_map_tools import convert_label_map

class Accuracy:
    def __init__(self, config, dt_annotation_path):
        label_map_path = os.path.join(config['base_dir'], config['label_map_file'])
        gt_annotation_path = os.path.join(config['base_dir'], config['annotation_file'])
        self.gt_annotations = self.load_annotation(gt_annotation_path)
        self.dt_annotations = self.load_annotation(dt_annotation_path)
        self.load_label_map(label_map_path)
        self.iou_threshold = config['iou_threshold']

    def load_label_map(self, label_map_path):
        self.attribute, self.label_id_to_name = load_label_map(label_map_path)
        self.label_name_to_id = convert_label_map(self.label_id_to_name)

    def load_annotation(self, annotation_path):
        with open(annotation_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def get_gt_annotation(self, anno_key):
        if self.gt_annotations is None:
            print("No annotation data")
        regions = self.gt_annotations[anno_key]['regions']
        return self.annotation_to_raw_data(regions)

    def get_dt_annotation(self, anno_key):
        if self.dt_annotations is None:
            print("No annotation data")
        regions = self.dt_annotations[anno_key]['regions']
        return self.annotation_to_raw_data(regions)

    def annotation_to_raw_data(self, regions):
        boxes = []
        label_ids = []
        for r_idx, attr in regions.items():
            s_attr = attr['shape_attributes']
            r_attr = attr['region_attributes']
            xmin = s_attr['x']
            ymin = s_attr['y']
            xmax = xmin + s_attr['height']
            ymax = ymin + s_attr['width']
            _name = r_attr['display_name']['0']
            _id = self.label_name_to_id[_name]
            boxes.append((xmin, ymin, xmax, ymax))
            label_ids.append(_id)
        return boxes, label_ids

    def iou_check(self, gt_boxes, gt_label_ids, dt_boxes, dt_label_ids):
        def get_iou(rect1, rect2):
            ''' get intersection over union
            '''
            xmin1, ymin1, xmax1, ymax1 = rect1
            xmin2, ymin2, xmax2, ymax2 = rect2
            x1 = max(xmin1, xmin2)
            y1 = max(ymin1, ymin2)
            x2 = min(xmax1, xmax2)
            y2 = min(ymax1, ymax2)
            inter_area = max(0, x2-x1+1) * max(0, y2-y1+1)
            rect1_area = (xmax1-xmin1+1) * (ymax1-ymin1+1)
            rect2_area = (xmax2-xmin2+1) * (ymax2-ymin2+1)
            total_area = float(rect1_area + rect2_area - inter_area)
            
            iou = inter_area / total_area if total_area != 0 else 0
            iou_box = (x1, y1, x2, y2)
            return iou_box, iou

        def find_max_iou_idx(ious, threshold, used_index):
                max_iou_idx = -1
                max_iou = -999
                for idx, iou in ious: # index, iou
                    if idx in used_index:
                        continue
                    if iou >= threshold:
                        if iou > max_iou or idx == 0 :
                            max_iou_idx, max_iou= idx, iou
                return max_iou_idx
        # Get Ground truth data
        gt_counter = Counter(gt_label_ids)
        dt_counter = Counter(dt_label_ids)
        gt_found = []
        used_index = []
        for gt_num, (gt_box, gt_label_id) in enumerate(zip(gt_boxes, gt_label_ids)): # gt_box loop
            dt_ious = []
            for dt_num, (dt_box, dt_label_id) in enumerate(zip(dt_boxes, dt_label_ids)): #  dt_box loop
                if gt_label_id == dt_label_id:
                    _, iou = get_iou(gt_box, dt_box) # iou: 0 ~ 1
                    dt_ious.append((dt_num, iou))
            # Return max iou dt box index. If there is no ious over threshod then return -1.
            max_iou_dt_idx = find_max_iou_idx(dt_ious, self.iou_threshold, used_index) 
            if max_iou_dt_idx > -1:
                gt_found.append(gt_label_id)
                used_index.append(max_iou_dt_idx) # Checed the dt box is used
        gt_found = Counter(gt_found)        
        return gt_counter, dt_counter, gt_found

    def precision_and_recall(self, label_name='all', print_raw_data=False):
        def print_result(gt_counter, dt_counter, gt_found, label_name='all'):
            print("{0:15}|{1:8}|{2:8}|{3:8}".format("Label_name"," GT_NUM"," DT_NUM"," GT_FOUND"))
            if label_name == 'all':
                for gt_label_id, gt_num in gt_counter.items():
                    gt_label_name = self.label_id_to_name[gt_label_id]
                    message = "{0:15}|{1:8}|{2:8}|{3:8}".format(
                        gt_label_name, gt_num, dt_counter[gt_label_id], gt_found[gt_label_id], )
                    print(message)
            else:
                label_id = self.label_name_to_id[label_name]
                message = "{0:15}|{1:8}|{2:8}|{3:8}".format(
                    label_name, gt_counter[label_id], dt_counter[label_id], gt_found[label_id])
                print(message)                    
        total_gt_counter = Counter()
        total_dt_counter = Counter()
        total_gt_found = Counter()
        example_cnt = 0
        for anno_key in self.gt_annotations.keys():
            try:
                dt_boxes, dt_label_ids = self.get_dt_annotation(anno_key)
            except KeyError as e:
                print("Key Error: No detection result of {}.".format(e))
                continue
            gt_boxes, gt_label_ids = self.get_gt_annotation(anno_key)
            gt_counter, dt_counter, gt_found = \
                self.iou_check(gt_boxes, gt_label_ids, dt_boxes, dt_label_ids)
            if print_raw_data:
                print("\nImage: {}".format(image_name))
                print_result(gt_counter, dt_counter, gt_found, label_name)
            total_gt_counter += gt_counter
            total_dt_counter += dt_counter
            total_gt_found   += gt_found
            example_cnt += 1
        if len(gt_counter) == 0:
            precision = 0
            recall = 0
        else:
            precision = sum(total_gt_found.values())/ float(sum(total_dt_counter.values()))
            recall = sum(total_gt_found.values())/ float(sum(total_gt_counter.values()))
        print("================================================================")
        if print_raw_data:
            print("Total")
            print_result(total_gt_counter, total_dt_counter, total_gt_found, label_name)
        print("Precision(%d, %s, %1.2f): %f"%(example_cnt, label_name, self.iou_threshold, precision))
        print("Recall(%d, %s, %1.2f): %f"%(example_cnt, label_name, self.iou_threshold, recall))
        return precision, recall