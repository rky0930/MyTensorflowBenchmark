# My Tensorflow Benchmark
This is My TensorFlow benchmark.  
All source code and environment are included in this repo.   
You can build your own benchmark tools and you can try to reproduce the results.  
Question/PR/Issues are welcome

### Result Screent Shot (2019/06/24)
<img src="screenshot/Screenshot%20from%202019-06-21%2019-49-30.png">

### Tested Language
 - Python
 - C/C++

### Tested CPU & GPU
 - Nvidia RTX2080 (8Gb)
 - GeForce GTX 1050 Mobile (4Gb)
 - AMD Ryzen 2700 
 - Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
 - 128-core NVIDIA Maxwell™ architecture-based GPU
 - Quad-core ARM® A57
 - ARM Cortex-A53 1.4GHz
 
### Tested Framework 
 - TensorFlow
 - TensorFlow-Lite

### Tested Archtecture + CPU/GPU + Framework + Language + Precision
 - x86_64 + TensorFlow + GPU + Python API + Float32
 - x86_64 + TensorFlow + CPU + Python API + Float32
 - x86_64 + TensorFlow + GPU + C API + Float32
 - x86_64 + TensorFlow + CPU + C API + Float32
 - x86_64 + TensorFlow-Lite + CPU + Python API + Float32
 - x86_64 + TensorFlow-Lite + CPU + Python API + Int8
 - aarch64 + TensorFlow + GPU + Python API + Float32
 - aarch64 + TensorFlow-Lite + CPU + Python API + Float32
 - aarch64 + TensorFlow-Lite + CPU + Python API + Int8
 - aarch64 + TensorFlow-Lite + CPU + C++ API + Float32
 - aarch64 + TensorFlow-Lite + CPU + C++ API + Int8
 - armv7l + TensorFlow + CPU + Python API + Float32
 - armv7l + TensorFlow-Lite + CPU + C++ API + Float32
 - armv7l + TensorFlow-Lite + CPU + C++ API + Int8
 
### Google spreadsheet
 - [google spreadsheet link](https://docs.google.com/spreadsheets/d/1c6aFzBUg2X9_EmMgAaPeV_Yn6-wrXbCGIaexmatnhi0/edit?usp=sharing)

### Benchmark tool
#### Run
 - [TensorFlow/TensorFlow-Lite - Python API](app/python) 
 - [TensorFlow/TensorFlow-Lite - C/C++ API](app/cc#tensorflow--tensorflow-lite-cc-api-benchmark-app)
 
#### Build
You can build all benchmark tool using the source code in this repo.
 - [TensorFlow/TensorFlow-Lite - Python API](app/python)  
 - [TensorFlow - C API](app/cc/src#tensorflow-c-api)
 - [TensorFlow-Lite - C++ API](app/cc/src#tensorflow-lite-aarch64armv7l-c-benchmark-tool)

### Dataset 
[MS-COCO Validataion 2017](dataset)


### Models
[SSD Mobilenet v2](models)

