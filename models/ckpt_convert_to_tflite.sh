#!/bin/bash
# Get more Information from:
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_on_mobile_tensorflowlite.md
#

# export TF_MODEL_RESEARCH_PATH= <TENSORFLOW_MODEL_PATH/research>
export TF_MODEL_RESEARCH_PATH=/home/yoon/workspace/models/research
export PYTHONPATH=$PYTHONPATH:$TF_MODEL_RESEARCH_PATH:$TF_MODEL_RESEARCH_PATH/slim
# export TF_SRC_PATH= <TENSORFLOW_SOURCE_CODE_PATH>
export TF_SRC_PATH=/home/yoon/workspace/tf_1_3_tflite_convert
export WORKSPACE=/media/yoon/hdd1/workspace/repos/rky0930/MyTensorflowBenchmark/models/object_detection # initalize MyTensorflowBenchmakr/models/objectdtecion path

###################
# Quantized model #
###################
MODEL_NAME=ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03
QT_BASE_DIR=${WORKSPACE}/tensorflow/${MODEL_NAME}
CONFIG_FILE=${QT_BASE_DIR}/pipeline.config
CHECKPOINT_PATH=${QT_BASE_DIR}/model.ckpt
MAX_DETECTIONS=50
OUTPUT_DIR=${WORKSPACE}/tensorflow_lite/${MODEL_NAME}_tflite_max${MAX_DETECTIONS}

# Create TensorFlow frozen graph with compatible ops that we can use with TensorFlow Lite
cd ${WORKSPACE}
python ${TF_MODEL_RESEARCH_PATH}/object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--max_detections=${MAX_DETECTIONS} \
--output_directory=${OUTPUT_DIR} \
--add_postprocessing_op=true

# Create optimized model by using TOCO(Tensorflow-lite Optimizing Converter)
cd ${TF_SRC_PATH}
# bazel run --config=opt tensorflow/lite/toco:toco -- \
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--allow_custom_ops

##################
# Floating model #
##################
MODEL_NAME=ssd_mobilenet_v2_coco_2018_03_29
FT_BASE_DIR=${WORKSPACE}/tensorflow/${MODEL_NAME}
CONFIG_FILE=${FT_BASE_DIR}/pipeline.config
CHECKPOINT_PATH=${FT_BASE_DIR}/model.ckpt
MAX_DETECTIONS=50
OUTPUT_DIR=${WORKSPACE}/tensorflow_lite/${MODEL_NAME}_tflite_max${MAX_DETECTIONS}

# Create TensorFlow frozen graph with compatible ops that we can use with TensorFlow Lite
cd ${WORKSPACE}
python ${TF_MODEL_RESEARCH_PATH}/object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--max_detections=${MAX_DETECTIONS} \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true

# Create optimized model by using TOCO(Tensorflow-lite Optimizing Converter)
cd ${TF_SRC_PATH}
# bazel run --config=opt tensorflow/lite/toco:toco -- \
bazel run -c opt tensorflow/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=FLOAT \
--allow_custom_ops
