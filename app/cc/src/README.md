### Build TensorFlow & TensorFlow-Lite C/C++ Benchmark Tool
#### Tensorflow C API
##### Prerequite
- Tensorflow C API
  - Detail is in [here](https://www.tensorflow.org/install/lang_c)
- OpenCV
- yaml-cpp
- jsoncpp

##### Build with docker
1. download docker image
```bash
# CPU
docker pull rky0930/c_api:opencv
nvidia-docker run -it -v <your_source_dir>:<your_source_dir> rky0930/c_api:opencv /bin/bash
# GPU
docker pull rky0930/c_api:cuda-10-cudnn7-opencv
docker run --runtime=nvidia -it -v <your_source_dir>:<your_source_dir> rky0930/c_api:cuda-10-cudnn7-opencv /bin/bash

```
2. Install yaml-cpp & jsoncpp
```bash
apt install libyaml-cpp-dev
apt install libjsoncpp-dev
```
3. Make
```bash
cd tensorflow_c_api/
make
ls bechmark # <- target binary
```


#### Tensorflow-Lite aarch64/armv7l C++ Benchmark Tool
##### Prerequite
- bazel
- opencv
- yaml-cpp
- jsoncpp

##### Build with docker
1. download docker image
```bash
docker pull rky0930/tflite:tf1.13.1-bazel0.19.2
docker run -it -v <your_source_dir>:<your_source_dir> rky0930/tflite:tf1.13.1-bazel0.19.2 /bin/bash
```
2. Make
```bash
# at MyTensorflowBenchmark directory
cd app/cc/src/tensorflow_lite/tensorflow/tensorflow/lite/tools/make/
# aarch64
./build_aarch64_lib.sh
ls gen/bechmark/aarch64_armv8-a/bin/benchmark # <- target binary

# armv7l
./build_rpi_lib.sh
ls gen/bechmark/rpi_armv7l/bin/benchmark # <- target binary
```
