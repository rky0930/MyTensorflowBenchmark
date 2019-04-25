import os
import glob
import cv2
import json
import traceback
import argparse
import numpy as np
import subprocess
import psutil
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
                    help="Benchmar Mode [fps|accuracy|memory_usage|all]. defult is all")
parser.add_argument("-p", "--print_acc_raw_data", default=False, action="store_true",
                    help="Print accuracy check raw data")
parser.add_argument("-s", "--save_inference_result", default=False, action="store_true",
                    help="Save inference result to annotation file")
parser.add_argument("-v", "--verbose", default=False, action="store_true",
                    help="verbose")
args = parser.parse_args()

config = load_config(args.config)
verbose = args.verbose
object_detection = None

def run_accuracy_check(accuracy):
    accuracy.precision_and_recall(
        label_name='all', 
        print_raw_data=args.print_acc_raw_data)

def run_memory_check(image_dir, max_example_num=10):
    example_cnt = 0
    total_rss = 0
    for image_path in sorted(glob.glob(image_dir)):
        if example_cnt >= max_example_num: 
            break
        boxes, socres, label_ids = object_detection.run(image_path)
        mem_info = psutil.Process().memory_full_info()
        total_rss += mem_info.rss
        example_cnt += 1
    avg_rss = total_rss/ example_cnt
    return example_cnt, avg_rss

def run_model_size_check():
    return object_detection.model_size

def run_object_detection(image_dir, accuracy, max_example_num):
    example_cnt = 0
    for image_path in sorted(glob.glob(image_dir)):
        if max_example_num and example_cnt >= max_example_num: 
            break
        if verbose and example_cnt % 100 == 0:
            print("image read: {}".format(example_cnt))
        boxes, socres, label_ids = object_detection.run(image_path)
        if accuracy:
            accuracy.add_raw_data(image_path, boxes, socres, label_ids)
        example_cnt += 1
    avg_inference_t = object_detection.get_average_inference_time()
    return example_cnt, avg_inference_t
    

if __name__ == "__main__":
    object_detection = \
        ObjectDetection(config['object_detection'], args.save_inference_result)
    mode = args.mode
    image_dir = os.path.join(config['benchmark']['base_dir'], config['benchmark']['image_dir'])
    if mode in ("memory_usage", "all"):
        example_cnt, avg_rss = run_memory_check(image_dir, config['memory_usage']['max_example_num'])
        avg_rss_mb = round(avg_rss/1024./1024.,3)
        mem_msg = "Average RSS for {} inference: {} Mb".format(example_cnt, avg_rss_mb)
    if mode in ("model_size", "all"):
        model_size = run_model_size_check()
        model_size_mb = round(model_size/1024./1024.,3)
        size_msg = "Model size: {}Mb".format(model_size_mb)
    if mode in ("fps", "all"):
        total_example_num, avg_inference_t = \
            run_object_detection(image_dir, None, config['fps']['max_example_num'])
        fps = round(1 / avg_inference_t, 3) if avg_inference_t != 0 else 0
        fps_msg = "Average_duration: {} FPS: {} for {} inference".format(avg_inference_t, fps, total_example_num)
    if mode in ("accuracy", "all"):
        accuracy = Accuracy(config['accuracy'])
        total_example_num, avg_inference_t = \
            run_object_detection(image_dir, accuracy, config['accuracy']['max_example_num'])
        run_accuracy_check(accuracy)
    print(mem_msg)
    print(size_msg)
    print(fps_msg)
    print("=== Benchmark END ===")
