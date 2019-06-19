if [ -z "$1" ] 
then 
    echo "Please set benchmark binary"
    echo "ex) ./run.sh bin/tensorflow_lite/aarch64/benchmark"
    exit
fi

#echo "----------------------TF FLOAT-------------------------"
#$1 -c configs/config_tf.yaml -m all
#python ../python/object_detection/accuracy_check.py -c=configs/config_tf.yaml -d=mscoco_val2017_inference_result.json
echo "----------------------TFLITE FLOAT-------------------------"
$1 -c configs/config_tflite.yaml -m all
python ../python/object_detection/accuracy_check.py -c=configs/config_tflite.yaml -d=mscoco_val2017_inference_result_tflite.json
echo "---------------------TFLITE INT8------------------------"
$1 -c configs/config_tflite_quantized.yaml -m all
python ../python/object_detection/accuracy_check.py -c=configs/config_tflite_quantized.yaml -d=mscoco_val2017_inference_result_tflite_quant.json
