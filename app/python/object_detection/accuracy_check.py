import argparse
from accuracy import Accuracy
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
parser.add_argument("-d", "--dt_anno_path", type=str, action="store", default="inference_result.json",
                    help="Detection result annotation file path")
args = parser.parse_args()
config = load_config(args.config)

if __name__ == "__main__":
    accuracy = Accuracy(config["accuracy"], args.dt_anno_path)
    _, _ = accuracy.precision_and_recall()