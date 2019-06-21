if [ -z "${1}" ];then 
    echo "Please set architecture"
    echo "ex) ./run.sh aarch64|armv7l"
    echo "ex) ./run.sh x86_64 cpu|gpu"
    exit
fi
if [ "$1" == "aarch64" ] || [ "$1" == "armv7l" ] ; then
    echo "----------------------TFLITE FLOAT-------------------------"
    bin/tensorflow_lite/${1}/benchmark -c configs/config_tflite.yaml -m all
    python ../python/accuracy_check.py -c=configs/config_tflite.yaml -d=mscoco_val2017_inference_result_tflite.json
    echo "---------------------TFLITE INT8------------------------"
    bin/tensorflow_lite/${1}/benchmark -c configs/config_tflite_quantized.yaml -m all
    python ../python/accuracy_check.py -c=configs/config_tflite_quantized.yaml -d=mscoco_val2017_inference_result_tflite_quant.json
fi
if [ "$1" == "x86_64" ] ; then
    if [ -z "${2}" ];then 
        echo "Please set accelerator"
        echo "ex) ./run.sh x86_64 cpu|gpu"
        exit
    fi
    echo "----------------------TF FLOAT-------------------------"
    bin/tensorflow/${1}/${2}/benchmark -c configs/config_tf.yaml -m all
    python ../python/object_detection/accuracy_check.py -c=configs/config_tf.yaml -d=mscoco_val2017_inference_result.json
fi

