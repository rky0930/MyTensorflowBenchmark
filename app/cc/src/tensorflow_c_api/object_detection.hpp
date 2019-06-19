#include <iostream>
#include <chrono> 
#include <cmath>
#include <opencv2/opencv.hpp>
#include "tensorflow/c/c_api.h"
#include "yaml-cpp/yaml.h"
#include "jsoncpp/json/json.h"
#include "utils/label_map_tools.hpp"

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
    
    TF_Graph* graph;
    TF_Buffer* graph_def;
    TF_ImportGraphDefOptions* graph_opts;
    TF_Status* graph_status;
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Status* sess_status;
    TF_Session* sess;  
    TF_Operation* input_op;
    TF_Output input_opout;
    std::vector<TF_Output> input_ops;
    std::vector<TF_Tensor*> input_values;
    TF_Operation* boxes_op;
    TF_Operation* scores_op;
    TF_Operation* classes_op;
    TF_Operation* num_detections_op;
    TF_Output boxes_opout, scores_opout, classes_opout, num_detections_opout;
    std::vector<TF_Output> output_ops;
    std::vector<TF_Tensor*> output_values;
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
    TF_Buffer* read_graph(std::string path);
    void DeleteInputValues();
    void ResetOutputValues();
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