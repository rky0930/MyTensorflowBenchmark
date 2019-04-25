### TensorFlow & TensorFlow-Lite Python API Benchmark App

'''
# python benchmark.py -h
usage: benchmark.py [-h] [-c CONFIG] [-m MODE] [-p] [-s] [-v]

Image Classification Benchmark Tool

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Configuration file path
  -m MODE, --mode MODE  Benchmar Mode [fps|accuracy|memory_usage|all]. defult
                        is all
  -p, --print_acc_raw_data
                        Print accuracy check raw data
  -s, --save_inference_result
                        Save inference result to annotation file
  -v, --verbose         verbose

    1) Benchmark FPS & Accuracy \
        python run.py -c=config.yaml \
    2) Benchmark FPS \
        python run.py -c=config.yaml -m=fps \
    2) Benchmark Accuracy \
        python run.py -c=config.yaml -m=accuracy
'''

### TesnsorFlow float32 benchmark
'''
python benchmark -c=config/config_tf.yaml 
'''
### TesnsorFlow-Lite float32 benchmark
'''
python benchmark -c=config/config_tflite.yaml 
'''
### TesnsorFlow-Lite uint8 benchmark
'''
python benchmark -c=config/config_tflite_quantized.yaml 
'''