import os
import glob
import cv2
import json
import traceback
import argparse
import numpy as np

from utils.config_tools import load_config
from object_detection import ObjectDetection
from utils.config_tools import load_label_map

arg_help_msg = """
    1) Benchmark FPS & Accuracy \\
        python run.py -c=config.yaml \\
    2) Benchmark FPS \\
        python run.py -c=config.yaml -m=fps \\
    2) Benchmark Accuracy \\
        python run.py -c=config.yaml -m=accuracy
"""
parser = argparse.ArgumentParser(description="Image Classification Benchmark Tool",
                                    epilog=arg_help_msg,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", type=str, action="store", default="config.yaml",
                    help="Configuration file path")
parser.add_argument("-m", "--mode", type=str, action="store", default="all",
                    help="Benchmar Mode [fps|accuracy|all]. defult is all")
args = parser.parse_args()

class Benchmark(object):
    def __init__(self, config, mode, object_detection):
        try:
            self.mode = mode
            self.image_dir = os.path.join(config['base_dir'], config['image_dir'])
            self.annotation_file = os.path.join(config['base_dir'], config['annotation_file'])
            self.max_example_num = float(config['max_example_num'])
            self.inference_result_file = None
            if self.mode in ('accuracy', 'all'):
                if 'inference_result_file' in config:
                    self.inference_result_file = config['inference_result_file']
                self.label_map_file = os.path.join(config['base_dir'], config['label_map'])      
                label_map = load_label_map(self.label_map_file)
                self.attribute = list(label_map.keys())[0]
                self.label_id_to_name = label_map[self.attribute]
        except KeyError:
            print("To check benchmark config file.")
            traceback.print_exc()
        self.object_detection = object_detection
        self.inference_result = {}

    def get_image_iterator(self):
        image_list = glob.glob(self.image_dir)
        for image_path in image_list:
            image_name = os.path.basename(image_path)
            image_size = os.path.getsize(image_path)
            image = cv2.imread(image_path)
            yield image_name, image_size, image

    def add_region_data(self, image_name, image_size, boxes, classes, image_height, image_width):   
        regions = {}
        for idx, (_box, _class) in enumerate(zip(boxes, classes)):
            try:
                ymin, ymax= _box[[0,2]] * image_height
                xmin, xmax= _box[[1,3]] * image_width
                height = ymax - ymin
                width = xmax - xmin
                print(_class.astype(int))
                obj_label_name = self.label_id_to_name[_class.astype(int)]
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
                            "0" : obj_label_name
                        }
                    }
                }
                regions[idx] = region
            except ValueError:
                continue 
        image_key = "{}{}".format(image_name, image_size)
        self.inference_result[image_key] = {
            "filename": image_name,
            "size": image_size,
            "regions": regions,
            "file_attributes": {}
        }     

    def run(self):
        total_duration_t = 0
        images = self.get_image_iterator()
        for idx, (image_name, image_size, image) in enumerate(images):
            print(image_name)
            if idx >= self.max_example_num: 
                break
            if idx % 100 == 0:
                print("image read: {}".format(idx, ))
            image_height, image_width, _ = image.shape
            duration_t, boxes, classes = self.object_detection.run(image)
            total_duration_t += duration_t
            if self.mode in ('accuracy', 'all'):
                self.add_region_data(image_name, image_size, boxes, classes, 
                                     image_height, image_width)

        average_duration_t = round(total_duration_t / float(idx), 3)
        fps = round(1/average_duration_t, 3)
        result_msg = "total example: {} average_duration: {} FPS: {}".format(idx, average_duration_t, fps)
        print(result_msg)
        if self.mode in ('accuracy', 'all'):
            
            if self.inference_result_file:
                with open(self.inference_result_file, "w") as ret_fid:
                    json.dump(self.inference_result, ret_fid, ensure_ascii=False)
                print("Inference result file is created: {}".format(self.inference_result))

if __name__ == "__main__":
    config = load_config(args.config)
    object_detection = ObjectDetection(config['object_detection'])
    benchmark = Benchmark(config['benchmark'], args.mode, object_detection) 
    benchmark.run()
