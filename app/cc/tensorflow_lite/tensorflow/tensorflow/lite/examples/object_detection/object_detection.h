#include <iostream>
#include <chrono> 
#include <cmath>
#include <opencv2/opencv.hpp>

#include "yaml-cpp/yaml.h"
#include "jsoncpp/json/json.h"
#include "utils/label_map_tools.h"

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/examples/object_detection/bitmap_helpers_impl.h"

struct Box {
  int xmin, ymin, xmax, ymax;
};

struct OD_Result_raw {
  float* boxes;
  float* scores;
  float* label_ids;
  float* num_detections;
};

struct OD_Result {
  std::vector<Box> boxes;
  std::vector<float> scores;
  std::vector<int> label_ids;
  int num_detections;
};


class ObjectDetection {
  private:
    std::string frozen_graph_path="";
    float confidence_score_threshold;
    int max_detections;
    long model_size = 0;
    std::string inference_result_file;
    std::string label_map_file_path;
    std::string attribute; 
    YAML::Node label_id_to_name; 

    std::unique_ptr<tflite::FlatBufferModel> graph;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::ops::builtin::BuiltinOpResolver resolver;

    Json::Value annotations;    
    std::chrono::duration<double> first_sess_run_elapsed_time;
    std::chrono::duration<double> total_sess_run_elapsed_time;
    int inference_counter = 0; 
    
    int verbose = 0;
    int visible = 0;
    int save_inference_result = 0;
  public:
    ObjectDetection(YAML::Node config, bool save_inference_result_flag);
    ~ObjectDetection() { close(); }
    void init();
    void set_graph();
    void set_label_map();
    void set_froze_graph_path(std::string path) { frozen_graph_path = path; }
    void set_verbose_flag(bool flag) { verbose = flag; }
    void set_visible_flag(bool flag) { visible = flag; }
    void preprocessing(IplImage* src, IplImage* dst);
    OD_Result_raw sess_run(IplImage* img);
    OD_Result run(const char* img_path, bool );
    OD_Result postprocessing(IplImage* src, OD_Result_raw od_result);
    void add_annotation(const char* img_path, OD_Result od_result);
    Json::Value raw_to_regions(OD_Result od_result);
    void write_to_file();
    double get_first_sess_run_elapsed_time();
    double get_average_sess_run_elapsed_time();
    int get_inference_counter();
    void reset_sess_run_elapsed_time();
    long get_model_size();
    int get_image_size(const char* img_path);
    void process_mem_usage(double& vm_usage, double& resident_set);
    void close();
};