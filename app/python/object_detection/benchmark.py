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

arg_help_msg = """
    1) Benchmark FPS \\
        python run.py -c=config.yaml \\
    2) Benchmark FPS \\
        python run.py -c=config.yaml -m=fps \\
"""
parser = argparse.ArgumentParser(description="Image Classification Benchmark Tool",
                                    epilog=arg_help_msg,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-c", "--config", type=str, action="store", default="config.yaml",
                    help="Configuration file path")
parser.add_argument("-m", "--mode", type=str, action="store", default="all",
                    help="Benchmar Mode [fps|model_size|memory_usage|all]. defult is all")
parser.add_argument("-v", "--verbose", default=False, action="store_true",
                    help="verbose")
args = parser.parse_args()

config = load_config(args.config)
verbose = args.verbose
mode    = args.mode
object_detection = None
    
def run_memory_check(image_dir, max_example_num=10):
    example_cnt = 0
    total_rss = 0
    for image_path in sorted(glob.glob(image_dir)):
        if verbose:
            print("image read: {}".format(example_cnt))
        if example_cnt >= max_example_num: 
            break
        boxes, socres, label_ids = object_detection.run(image_path, False)
        mem_info = psutil.Process().memory_full_info()
        total_rss += mem_info.rss
        example_cnt += 1
    avg_rss = total_rss/ example_cnt
    return example_cnt, avg_rss

def run_model_size_check():
    return object_detection.model_size

def run_object_detection(image_dir, max_example_num, save_inference_result_flag):
    example_cnt = 0
    for image_path in sorted(glob.glob(image_dir)):
        if max_example_num and example_cnt >= max_example_num: 
            break
        if verbose:
            print("image read: {}".format(example_cnt))
        boxes, socres, label_ids = object_detection.run(image_path, save_inference_result_flag)
        example_cnt += 1
    avg_inference_t = object_detection.get_average_inference_time()
    first_inference_t = object_detection.get_first_inference_time()
    return example_cnt, first_inference_t, avg_inference_t
    

if __name__ == "__main__":
    save_inference_result_flag = True if mode in ("save_inference_result", "all") else False
    object_detection = \
        ObjectDetection(config['object_detection'], save_inference_result_flag, verbose)
    image_dir = os.path.join(config['benchmark']['base_dir'], config['benchmark']['image_dir'])
    if mode in ("memory_usage", "all"):
        print("==Start memory usage check==")
        example_cnt, avg_rss = run_memory_check(image_dir, config['memory_usage']['max_example_num'])
        avg_rss_mb = round(avg_rss/1024./1024.,3)
        mem_msg = "Average RSS for {} inference: {} Mb".format(example_cnt, avg_rss_mb)
        if not verbose:
            print(mem_msg)
        print("==End memory usage check==")
    if mode in ("model_size", "all"):
        print("==Start model size check==")
        model_size = run_model_size_check()
        model_size_mb = round(model_size/1024./1024.,3)
        size_msg = "Model size: {}Mb".format(model_size_mb)
        if not verbose:
            print(size_msg)
        print("==End model size check==")
    if mode in ("fps", "all"):
        print("==Start FPS check==")
        total_example_num, first_inference_t, avg_inference_t = \
            run_object_detection(image_dir, config['fps']['max_example_num'], False)
        fps = round(1 / avg_inference_t, 3) if avg_inference_t != 0 else 0
        fps_msg = "first_inference_t: {} sec | avg_inference_t: {} sec | Avg FPS: {} for {}~{} inference".format(
            first_inference_t, avg_inference_t, fps, 2, total_example_num)
        if not verbose:
            print(fps_msg)
        print("==End FPS check==")
    if mode in ("save_inference_result", "all"):
        print("==Start Save Inference result==")
        example_cnt, _, _ = run_object_detection(image_dir, 
            config['object_detection']['save_inference_result']['max_example_num'], True)
        object_detection.write_to_file()
        sv_msg = "Saved {} examples inference result.".format(example_cnt)
        if not verbose:
            print(sv_msg)
        print("==End Save Annotation==")
    if verbose:
        print(mem_msg)
        print(size_msg)
        print(fps_msg)
        print(sv_msg)
    print("=== Benchmark END ===")
