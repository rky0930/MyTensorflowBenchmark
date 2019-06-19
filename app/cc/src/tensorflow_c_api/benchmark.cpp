#include <iostream>
#include <getopt.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include "utils/config_tools.hpp"
#include "object_detection.hpp"

void help(char *argv);
void run_object_detection(ObjectDetection &object_detection, std::string image_path, int max_example_num);
void run_memory_check(ObjectDetection &object_detection, std::string image_path, int max_example_num);
void close(const char *s);

int verbose_flag = 0;
int visible_flag = 0;
int save_inference_result_flag = 0;

void help(char *argv) {
    std::cout<<"Usage : "<<argv<<" --(f)rozen_graph_path --(i)mage_path [--(c)onfidence_score_threshold] [--(m)ax_detections] [--(v)erbose] [--(s)how]"<<std::endl;
    exit(0);
}

void run_object_detection(ObjectDetection &object_detection, std::string image_path, int max_example_num, bool save_inference_result_flag) {
  struct stat st;
  if (stat(image_path.c_str(), &st) < 0) close("stat");
  if (!S_ISDIR(st.st_mode)) {
    // File
    std::cout<<image_path<<std::endl;
    object_detection.run(image_path.c_str(), save_inference_result_flag);
  }else{
    // Directory
    DIR *d;
    struct dirent *ent;
    d = opendir(image_path.c_str());
    int example_cnt = 0;
    while (ent = readdir(d)) {
      if (example_cnt >= max_example_num) break;
      if (strcmp(ent->d_name, ".") == 0) continue;
      if (strcmp(ent->d_name, "..") == 0) continue;
      std::string full_path = image_path;
      if (full_path.back() != '/') {
        full_path.append("/");
      }
      full_path.append(ent->d_name);
      if (verbose_flag) {
        std::cout<<full_path<<std::endl;
      }
      object_detection.run(full_path.c_str(), save_inference_result_flag);
      example_cnt++;
    }
  }
  int inference_counter = object_detection.get_inference_counter();
  double first_sess_run_elapsed_time;
  double average_sess_run_elapsed_time;
  if (inference_counter == 0) {
    std::cout<<"No data to check FPS"<<std::endl;
    return;
  } 
  if (inference_counter > 0) {
    first_sess_run_elapsed_time = object_detection.get_first_sess_run_elapsed_time();
    std::cout<<"First Session Run Elapsed time is "<<first_sess_run_elapsed_time<<" seconds"<<std::endl;
  }
  if (inference_counter > 1) {
    average_sess_run_elapsed_time = object_detection.get_average_sess_run_elapsed_time();
    std::cout<<"Average Session Run Elapsed time is "<<average_sess_run_elapsed_time<<" seconds "\
        "for "<<inference_counter-1<<" examples"<<std::endl;
  }
}

void run_memory_check(ObjectDetection &object_detection, std::string image_path, int max_example_num) {
  struct stat st;
  double total_vm, total_rss;
  float average_vm, average_rss;
  double vm, rss;
  int example_cnt = 0;
  std::cout<<image_path<<std::endl;
  if (stat(image_path.c_str(), &st) < 0) close("stat");
  if (!S_ISDIR(st.st_mode)) {
    // File
    std::cout<<image_path<<std::endl;
    object_detection.run(image_path.c_str(), false);
    object_detection.process_mem_usage(vm, rss);
    total_vm = total_vm + vm;
    total_vm = total_rss + rss;
    example_cnt++;
  }else{
    // Directory
    DIR *d;
    struct dirent *ent;
    d = opendir(image_path.c_str());
    while (ent = readdir(d)) {
      if (example_cnt >= max_example_num) break;
      if (strcmp(ent->d_name, ".") == 0) continue;
      if (strcmp(ent->d_name, "..") == 0) continue;
      std::string full_path = image_path;
      if (full_path.back() != '/') {
        full_path.append("/");
      }
      full_path.append(ent->d_name);
      if (verbose_flag) {
        std::cout<<full_path<<std::endl;
      }
      object_detection.run(full_path.c_str(), false);
      object_detection.process_mem_usage(vm, rss);
      total_vm = total_vm + vm;
      total_rss = total_rss + rss;
      example_cnt++;
    }
  }
  average_vm = (float)total_vm / example_cnt;
  average_rss = (float)total_rss / example_cnt / 1024.0;
  // std::cout<<"Average VM: "<<average_vm<<" Mb, Average RSS: "<<average_rss<<" Mb"<<std::endl;
  std::cout<<"Average RSS for "<<example_cnt<<" inference: "<<average_rss<<" Mb"<<std::endl;
}

void close(const char *s) {
    perror(s);
    exit(1);
}

int main (int argc, char **argv)
{

  std::string config_path = "";
  std::string mode = "";
  int c;
  while (1) {
    static struct option long_options[] =
      {
        {"config_path", required_argument, 0, 'c'},
        {"mode", required_argument, 0, 'm'},          
        {"verbose", no_argument, 0, 'v'},
        {"show", no_argument, 0, 's'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
      };
    /* getopt_long stores the option index here. */
    int option_index = 0;
    c = getopt_long (argc, argv, "c:m:vsrh",
                      long_options, &option_index);
    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c)
    {
      case 'c':
        config_path = std::string(optarg);
        break;
      case 'm':
        mode = std::string(optarg);
        std::cout<<"Mode: "<<mode<<std::endl;
        break;
      case 'v':
        verbose_flag = 1;
        break;
      case 's':
        visible_flag = 1;
        break;
      case 'h':
      case '?':
      default:
        help(argv[0]);
    }
  }
    if (config_path == "") {
      std::cerr<<"Please set -(-c)onfig_path"<<std::endl;
      exit(1);
    }
    if (mode == "") {
      std::cerr<<"Please set -(-m)ode"<<std::endl;
      exit(1);
    }
    YAML::Node config = load_config(config_path);
    std::string base_dir = config["benchmark"]["base_dir"].as<std::string>();
    std::string image_dir = config["benchmark"]["image_dir"].as<std::string>();
    std::string last_ch = image_dir.substr(image_dir.size()-1, image_dir.size());
    if (last_ch == "*") {
      image_dir = image_dir.substr(0, image_dir.size()-1);
    }
    std::string image_dir_path = base_dir + "/" + image_dir;
    if (mode == "all" or mode == "save_inference_result") {
      save_inference_result_flag = true;
    }else {
      save_inference_result_flag = false;
    }
    ObjectDetection object_detection(config["object_detection"], 
                                     save_inference_result_flag);
    object_detection.set_verbose_flag(verbose_flag);
    object_detection.set_visible_flag(visible_flag);
    object_detection.init();
    int max_example_num;
    if (mode == "memory_usage" or mode == "all") {
      std::cout<<"==Start memory usage check=="<<std::endl;
      max_example_num = config["memory_usage"]["max_example_num"].as<int>();
      run_memory_check(object_detection, image_dir_path, max_example_num);
    }
    if (mode == "model_size" or mode == "all") {
      long model_size = object_detection.get_model_size();
      std::cout<<"Model size: "<<(float)model_size /1024.0/ 1024.0<<" Mb"<<std::endl;
    }
    if (mode == "fps" or mode == "all") {
      std::cout<<"==Start FPS check=="<<std::endl;
      max_example_num = config["fps"]["max_example_num"].as<int>();
      run_object_detection(object_detection, image_dir_path, max_example_num, false);
    }
    if (mode == "save_inference_result" or mode == "all") {
      std::cout<<"==Start Save Inference result=="<<std::endl;
      max_example_num = \
        config["object_detection"]["save_inference_result"]["max_example_num"].as<int>();
      run_object_detection(object_detection, image_dir_path, max_example_num, true);
      object_detection.write_to_file();
    }
    
    return 0;
}
