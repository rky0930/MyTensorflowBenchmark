import argparse
import os
from libs.accuracy import Accuracy
from utils.config_tools import load_config

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
parser.add_argument("-d", "--od_result_file", type=str, action="store", default="inference_result.json",
                    help="Detection result annotation file path")
args = parser.parse_args()
config = load_config(args.config)

if __name__ == "__main__":
    print("= Start Accration Check = ")
    base_dir = config["accuracy"]["base_dir"]
    anno_file_path = os.path.join(base_dir, config["accuracy"]["annotation_file"])
    print(" - iou_threshold: {}".format(config["accuracy"]["iou_threshold"]))
    print(" - Ground Truth File: {}".format(anno_file_path))
    print(" - OD Result File: {}".format(args.od_result_file))
    accuracy = Accuracy(config["accuracy"], args.od_result_file)

    _, _ = accuracy.precision_and_recall()