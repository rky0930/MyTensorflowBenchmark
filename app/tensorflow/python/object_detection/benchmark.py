import os
import glob
import cv2
import json
import traceback
import argparse
import numpy as np

from utils.config_tools import load_config
from object_detection import ObjectDetection
from accuracy import Accuracy

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
    def __init__(self, config, mode, object_detection, accuracy):
        try:
            self.mode = mode
            self.image_dir = os.path.join(config['base_dir'], config['image_dir'])
            self.max_example_num = float(config['max_example_num'])

        except KeyError:
            print("To check benchmark config file.")
            traceback.print_exc()
        self.object_detection = object_detection
        self.accuracy = accuracy

    def get_image_iterator(self):
        image_list = glob.glob(self.image_dir)
        for image_path in image_list:
            image_name = os.path.basename(image_path)
            image_size = os.path.getsize(image_path)
            image = cv2.imread(image_path)
            yield image_name, image_size, image

    def run(self):
        image_iter = self.get_image_iterator()
        for idx, (image_name, image_size, image) in enumerate(image_iter):
            if idx >= self.max_example_num: 
                break
            if idx % 100 == 0:
                print("image read: {}".format(idx, ))
            boxes, socres, label_ids = self.object_detection.run(image)
            if self.mode in ('accuracy', 'all'):
                accuracy.add_raw_data(image_name, image_size, boxes, socres, label_ids)
        # Print Result
        average_duration_t = self.object_detection.get_average_inference_time()
        fps = self.object_detection.get_fps()
        result_msg = "total example: {} average_duration: {} FPS: {}".format(idx, average_duration_t, fps)
        print(result_msg)
        accuracy.average_precision(iou_threshold=0.1, label_name='all', verbose=True, print_ap_raw_data=False)
        accuracy.write_to_file()

if __name__ == "__main__":
    config = load_config(args.config)
    print("1. load_config success")
    object_detection = ObjectDetection(config['object_detection'])
    print("2. ObjectDetection init success")
    accuracy = Accuracy(config['accuracy'])
    print("3. Accuracy init success")
    benchmark = Benchmark(config['benchmark'], args.mode, object_detection, accuracy) 
    print("4. Benchmark init success")
    benchmark.run()
