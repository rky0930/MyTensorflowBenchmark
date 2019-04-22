import os
import json
import yaml
import numpy as np
from collections import defaultdict

class Accuracy:
    def __init__(self, config):
        self.config = config
        self.load_annotation()        
        self.load_label_map()
        self.iou_threshold = config['iou_threshold']
        self.raw_data_list = []
        self.inference_result_file = None
        if 'inference_result_file' in config:
            self.inference_result_file = config['inference_result_file']

    def add_raw_data(self, image_name, image_size, boxes, socres, label_ids):
        raw_data = (image_name, image_size, boxes, socres, label_ids)
        self.raw_data_list.append(raw_data)

    def load_label_map(self):
        def convert_label_map(label_map):
            if not isinstance(label_map, dict):
                raise ValueError("label map is not dictionary type")
            return dict((_name,_id) for _id, _name in label_map.items())
        label_map_file = os.path.join(
            self.config['base_dir'], self.config['label_map'])
        with open(label_map_file, 'r') as stream:
            try:
                label_map = yaml.safe_load(stream)
                self.attribute = list(label_map.keys())[0]
                self.label_id_to_name = label_map[self.attribute]
                self.label_name_to_id = convert_label_map(self.label_id_to_name)
            except yaml.YAMLError as exc:
                print(exc)
        
    def load_annotation(self):
        annotation_file = os.path.join(
            self.config['base_dir'], self.config['annotation_file'])
        with open(annotation_file, 'r') as stream:
            try:
                self.annotations = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def get_gt_raw_data(self, image_name, image_size):
        if self.annotations is None:
            print("No annotation data")
        anno_key = "{}{}".format(image_name, image_size)
        regions = self.annotations[anno_key]['regions']
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

    def create_matrices(self, gt_boxes, gt_label_ids, dt_boxes, dt_label_ids, iou_threshold):
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
            iou = inter_area / float(rect1_area + rect2_area - inter_area)
            iou_box = (x1, y1, x2, y2)
            return iou_box, iou

        gt_counter = len(gt_label_ids)
        dt_counter = len(dt_label_ids)
        iou_matrices = defaultdict(list)
        used_index = []
        for gt_num, (gt_box, gt_label_id) in enumerate(zip(gt_boxes, gt_label_ids)): # gt_box loop
            gt_label_name = self.label_id_to_name[gt_label_id]
            dt_ious = defaultdict(dict)
            for dt_num, (dt_box, dt_label_id) in enumerate(zip(dt_boxes, dt_label_ids)): #  dt_box loop
                _, iou = get_iou(gt_box, dt_box)
                if gt_label_id == dt_label_id:
                    dt_ious[gt_label_name][dt_num] = iou
                else:
                    dt_ious[gt_label_name][dt_num] = -1 # -1 is wrong category
            iou_matrices[gt_label_name].append(dt_ious[gt_label_name]) # matrix by category
        return iou_matrices, gt_counter, dt_counter

    def average_precision(self, iou_threshold=0.5, label_name='all', verbose=False, print_ap_raw_data=False):
        def optimal_box_index(dt_ious_of_one_category, threshold, used_index):
            max_index = -1
            max_value = -999
            for i, v in dt_ious_of_one_category.items(): # index, iou
                if i in used_index:
                    continue
                if v >= threshold:
                    if v > max_value or i == 0 :
                        max_index, max_value= i, v
            return max_index
        total_gt_counter = 0
        total_gt_found = 0
        total_under_iou_threshold = 0
        for dt_num, (image_name, image_size, dt_boxes, dt_socres, dt_label_ids) in enumerate(self.raw_data_list):
            gt_found = 0
            gt_not_found = 0
            gt_boxes, gt_label_ids = self.get_gt_raw_data(image_name, image_size)
            iou_matrices, gt_counter, dt_counter = \
                    self.create_matrices(gt_boxes, gt_label_ids, dt_boxes, dt_label_ids, iou_threshold)
            for matrix_num, (gt_label_name, matrix) in enumerate(iou_matrices.items()):
                # If you want to get  AP of specific label. ex) AP of persson label
                if gt_label_name == label_name or label_name == 'all': 
                    used_index = []
                    for gt_num, dt_ious in enumerate(matrix): # over iou of the gt and all dts
                        dt_ious_values = np.array(list(dt_ious.values()))  # 1. over iou thres hold, 2. under iou thres hold 3. under 0: wrong classification 
                        if gt_num == 0 and matrix_num == 0:                                        
                            under_iou_threshold_index                          = np.logical_and( dt_ious_values < iou_threshold, dt_ious_values >= 0)
                            under_iou_threshold_and_wrong_classification_index = np.logical_and(-dt_ious_values < iou_threshold, dt_ious_values < 0)
                            wrong_classification_index                         = -dt_ious_values > iou_threshold                            
                        else:
                            new_under_iou_threshold_index                          = np.logical_and( dt_ious_values < iou_threshold, dt_ious_values >= 0)
                            new_under_iou_threshold_and_wrong_classification_index = np.logical_and(-dt_ious_values < iou_threshold, dt_ious_values < 0)
                            new_wrong_classification_index                         = -dt_ious_values > iou_threshold                            
                            under_iou_threshold_index                          = np.logical_and(under_iou_threshold_index                          , new_under_iou_threshold_index                          ) 
                            under_iou_threshold_and_wrong_classification_index = np.logical_and(under_iou_threshold_and_wrong_classification_index , new_under_iou_threshold_and_wrong_classification_index ) 
                            wrong_classification_index                         = np.logical_and(wrong_classification_index                         , new_wrong_classification_index                         ) 
                        op_index = optimal_box_index(dt_ious, iou_threshold, used_index) # If there is mone than one box over IoU threshod, select one 
                        if op_index > -1:
                            gt_found += 1
                            used_index.append(op_index) # Checed the dt box is used
                    #print(under_iou_threshold_index, iou_threshold, sum(under_iou_threshold_index))
            under_iou_threshold = sum(under_iou_threshold_index)
            wrong_classification = sum(wrong_classification_index)
            under_iou_threshold_and_wrong_classification = sum(under_iou_threshold_and_wrong_classification_index)
            duplicated_gt_found = dt_counter-gt_found-under_iou_threshold-under_iou_threshold_and_wrong_classification-wrong_classification
            gt_not_found = gt_counter-gt_found
            if print_ap_raw_data:
                if dt_num == 0: # print Header
                    message = "Image Name\tTrue PeopleCount\tPredicted People Count\tDetect Success({} IoU:{})" \
                            "\tDetect Faile\tUnder Iou Threshold\tDuplicated Box Matching\n".format(label_name, iou_threshold)
                    print(message)
                # message = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
                #     image_name,  gt_counter, dt_counter, gt_found, gt_not_found, under_iou_threshold, duplicated_gt_found)
                # print(message)
                message = """{}\n
                    gt_counter:{}\n
                    dt_counter:{}\n
                    gt_found:{}\n
                    gt_not_found:{}\n
                    under_iou_threshold:{}\n
                    wrong_classification:{}\n
                    under_iou_threshold_and_wrong_classification:{}\n
                    duplicated_gt_found:{}\n""".format(
                    image_name, gt_counter, dt_counter, gt_found, gt_not_found, under_iou_threshold, 
                    wrong_classification, under_iou_threshold_and_wrong_classification, duplicated_gt_found)
                print(message)
            print(dt_counter)
            print((gt_found + under_iou_threshold + \
                wrong_classification + under_iou_threshold_and_wrong_classification + duplicated_gt_found))
            assert under_iou_threshold_and_wrong_classification == 0
            assert dt_counter == (gt_found + under_iou_threshold + \
                wrong_classification + under_iou_threshold_and_wrong_classification + duplicated_gt_found)
                
            total_gt_counter += gt_counter
            total_gt_found += gt_found
            total_under_iou_threshold += under_iou_threshold
        if gt_counter == 0:
            ap = 0
        else:
            ap = total_gt_found / float(total_gt_counter)
        if verbose:
            print("AP(%s, %1.2f): %f\n"%(label_name, iou_threshold, ap))
        return ap

    def raw_to_regions(self, boxes, label_ids):
        regions = {}
        for idx, (box, label_id) in enumerate(zip(boxes, label_ids)):
            try:
                xmin, ymin, xmax, ymax = box
                height = ymax - ymin
                width = xmax - xmin
                label_name = self.label_id_to_name[label_id.astype(int)]
                region = {
                    "shape_attributes": {
                        "name": "rect",
                        "x": int(round(xmin)),
                        "y": int(round(ymin)),
                        "width":  int(round(width)),
                        "height": int(round(height))
                    },
                    "region_attributes": {
                        self.attribute: {
                            "0" : label_name
                        }
                    }
                }
                regions[idx] = region
            except ValueError:
                continue 
        return regions    

    def write_to_file(self):
        inference_result = {}
        for image_name, image_size, boxes, socres, label_ids in self.raw_data_list:
            regions = self.raw_to_regions(boxes, label_ids)
            anno_key = "{}{}".format(image_name, image_size)
            inference_result[anno_key] = {
                "filename": image_name,
                "size": image_size,
                "regions": regions,
                "file_attributes": {}
            }
        with open(self.inference_result_file, "w") as ret_fid:
            json.dump(inference_result, ret_fid, ensure_ascii=False)
        print("Inference result file is created: {}".format(self.inference_result_file))