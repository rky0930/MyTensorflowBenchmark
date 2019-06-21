echo "----------------------TF FLOAT-------------------------"
python benchmark.py -c=configs/config_tf.yaml
python accuracy_check.py -c=configs/config_tf.yaml -d=mscoco_val2017_inference_result.json
echo "----------------------TFLITE FLOAT-------------------------"
python benchmark.py -c=configs/config_tflite.yaml
python accuracy_check.py -c=configs/config_tflite.yaml -d=mscoco_val2017_inference_result_tflite.json
echo "---------------------TFLITE INT8------------------------"
python benchmark.py -c=configs/config_tflite_quantized.yaml
python accuracy_check.py -c=configs/config_tflite_quantized.yaml -d=mscoco_val2017_inference_result_tflite_quant.json
