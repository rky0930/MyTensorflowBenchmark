#include "object_detection.h"

ObjectDetection::ObjectDetection(YAML::Node config, bool save_inference_result_flag) {
  std::string base_dir = config["base_dir"].as<std::string>();
  std::string checkpoint = config["checkpoint"].as<std::string>();
  this->frozen_graph_path = base_dir + "/" + checkpoint;
  this->confidence_score_threshold = config["confidence_score_threshold"].as<float>();
  this->max_detections = config["save_inference_result"]["max_example_num"].as<int>();
  this->save_inference_result = save_inference_result_flag;
  reset_sess_run_elapsed_time();
  if (this->save_inference_result) {
    this->inference_result_file = config["save_inference_result"]["file"].as<std::string>();
  }
  std::string label_map_file = config["label_map_file"].as<std::string>();
  this->label_map_file_path = base_dir + "/" + label_map_file;
}

void ObjectDetection::init() {
  this->set_graph();
  this->set_label_map();
}

void ObjectDetection::set_label_map() {
  std::string attribute; 
  YAML::Node label_id_to_name; 
  load_label_map(this->label_map_file_path, attribute, label_id_to_name);
  this->attribute = attribute;
  this->label_id_to_name = label_id_to_name;  
}

void ObjectDetection::set_graph() {
  std::cout<<"Load Model: "<<this->frozen_graph_path<<std::endl;
  this->graph = tflite::FlatBufferModel::BuildFromFile(this->frozen_graph_path.c_str());
  if (!this->graph) {
    std::cout <<"Failed to mmap model: "<<this->frozen_graph_path<<std::endl;
    exit(-1);
  }
  this->graph->error_reporter();
  std::cout<<"resolved reporter"<<std::endl;;
  tflite::InterpreterBuilder(*this->graph, this->resolver)(&this->interpreter);
  if (!this->interpreter) {
    std::cout<<"Failed to construct interpreter"<<std::endl;
    exit(-1);
  }

  if (this->verbose) {
    std::cout << "tensors size: " << this->interpreter->tensors_size() << "\n";
    std::cout << "nodes size: " << this->interpreter->nodes_size() << "\n";
    std::cout << "inputs: " << this->interpreter->inputs().size() << "\n";
    std::cout << "input(0) name: " << this->interpreter->GetInputName(0) << "\n";

    int t_size = this->interpreter->tensors_size();
    int total_size = 0;
    for (int i = 0; i < t_size; i++) {
      if (this->interpreter->tensor(i)->name) {
        std::cout << i << ": " << this->interpreter->tensor(i)->name << ", "
                  << this->interpreter->tensor(i)->bytes << ", "
                  << this->interpreter->tensor(i)->type << ", "
                  << this->interpreter->tensor(i)->params.scale << ", "
                  << this->interpreter->tensor(i)->params.zero_point << "\n";
        total_size = total_size + this->interpreter->tensor(i)->bytes;
      }
    }
    std::cout <<"total_size: "<< total_size << "\n";
  }

  // if (s->number_of_threads != -1) {
  //   this->interpreter->SetNumThreads(s->number_of_threads);
  // }
}

void ObjectDetection::preprocessing(IplImage* src, IplImage* dst) {
  cvCvtColor(src, dst, CV_BGR2RGB);
}

OD_Result_raw ObjectDetection::sess_run(IplImage* img) {
  int img_width = img->width;
  int img_height = img->height;
  int img_channel = img->nChannels;  

  int input = this->interpreter->inputs()[0];  
  if (this->verbose) std::cout << "input: " << input << "\n";
  const std::vector<int> inputs = this->interpreter->inputs();
  const std::vector<int> outputs = this->interpreter->outputs();
  if (this->verbose) {
      std::cout << "number of inputs: " << inputs.size() << "\n";
      std::cout << "number of outputs: " << outputs.size() << "\n";
  }

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cout << "Failed to allocate tensors!";
  }

  if (this->verbose) PrintInterpreterState(interpreter.get());
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int input_height = dims->data[1];
  int input_width = dims->data[2];
  int input_channels = dims->data[3];

  switch (interpreter->tensor(input)->type) {
    case kTfLiteFloat32:
      tflite::object_detection::resize<float>(interpreter->typed_tensor<float>(input), (uint8_t*)img->imageData,
                    img_height, img_width, img_channel, input_height,
                    input_width, input_channels, true);
      break;
    case kTfLiteUInt8:
      tflite::object_detection::resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), (uint8_t*)img->imageData,
                      img_height, img_width, img_channel, input_height,
                      input_width, input_channels, false);
      break;
    default:
      std::cout << "cannot handle input type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }

  // profiling::Profiler* profiler = new profiling::Profiler();
  // interpreter->SetProfiler(profiler);
  // if (s->profiling) profiler->StartProfiling();

  // INVOKE !!
  std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
  start_time = std::chrono::system_clock::now(); 

  if (interpreter->Invoke() != kTfLiteOk) {
    std::cout << "Failed to invoke tflite!\n";
  }
  end_time = std::chrono::system_clock::now();
  if (this->inference_counter == 0) {
    this->first_sess_run_elapsed_time = (end_time - start_time);
  }else{
    this->total_sess_run_elapsed_time = this->total_sess_run_elapsed_time + (end_time - start_time);
  }
  this->inference_counter++;
  // if (s->profiling) {
  //   profiler->StopProfiling();
  //   auto profile_events = profiler->GetProfileEvents();
  //   for (int i = 0; i < profile_events.size(); i++) {
  //     auto op_index = profile_events[i]->event_metadata;
  //     const auto node_and_registration =
  //         interpreter->node_and_registration(op_index);
  //     const TfLiteRegistration registration = node_and_registration->second;
  //     PrintProfilingInfo(profile_events[i], op_index, registration);
  //   }
  // }
  std::vector<std::pair<float, int>> top_results;
  int boxes_idx   = interpreter->outputs()[0];
  int classes_idx = interpreter->outputs()[1];
  int scores_idx  = interpreter->outputs()[2];
  int num_det_idx = interpreter->outputs()[3];
  TfLiteIntArray* box_dims     = interpreter->tensor(boxes_idx)->dims;
  TfLiteIntArray* class_dims   = interpreter->tensor(classes_idx)->dims;
  TfLiteIntArray* score_dims   = interpreter->tensor(scores_idx)->dims;
  TfLiteIntArray* num_det_dims = interpreter->tensor(num_det_idx)->dims;
  OD_Result_raw od_result_raw;
  od_result_raw.boxes = interpreter->typed_output_tensor<float>(0);
  od_result_raw.label_ids = interpreter->typed_output_tensor<float>(1);
  od_result_raw.scores = interpreter->typed_output_tensor<float>(2);
  od_result_raw.num_detections = interpreter->typed_output_tensor<float>(3);
  return od_result_raw;
}

OD_Result ObjectDetection::run(const char* img_path, bool save_inference_result_flag) {
  IplImage* img = cvLoadImage(img_path, CV_LOAD_IMAGE_COLOR);
  if (!img)
  {
    std::cout<<"Image load failed: "<<img_path<<std::endl;
    exit(1);
  }
  this->preprocessing(img, img);
  if (this->verbose) {
    std::cout<<"First pixel: (";
    std::cout<<unsigned((uint8_t)img->imageData[0])<<", ";
    std::cout<<unsigned((uint8_t)img->imageData[1])<<", ";
    std::cout<<unsigned((uint8_t)img->imageData[2])<<")"<<std::endl;;
    std::cout<<"img size: "<<img->width<<", "<<img->height<<", "<<img->nChannels<<std::endl;
    std::cout<<"img->depth: "<<img->depth<<std::endl;
    std::cout<<"img->imgSize: "<<img->imageSize<<std::endl;
    std::cout<<"img->width: "<<img->width<<std::endl;
    std::cout<<"img->height: "<<img->height<<std::endl;
    std::cout<<"img->nCHannels: "<<img->nChannels<<std::endl;
    std::cout<<"img->alphaChannel: "<<img->alphaChannel<<std::endl;
    std::cout<<"img->channelSeq : "<<img->channelSeq <<std::endl;
    std::cout<<"img->colorModel: "<<img->colorModel<<std::endl;
    std::cout<<"img->dataOrder: "<<img->dataOrder<<std::endl;
    std::cout<<"img->nSize: "<<img->nSize<<std::endl;
    std::cout<<"img->widthStep: "<<img->widthStep<<std::endl;
  }
  OD_Result_raw od_result_raw;
  OD_Result od_result;
  od_result_raw = this->sess_run(img);
  od_result = this->postprocessing(img, od_result_raw);
  cvReleaseImage(&img);
  if (save_inference_result_flag) {
    this->add_annotation(img_path, od_result);
  }
  return od_result;
}

OD_Result ObjectDetection::postprocessing(IplImage* img, OD_Result_raw od_result_raw) {
  int img_width = img->width;
  int img_height = img->height;
  int img_channel = img->nChannels;  
  int num_detections = (int)od_result_raw.num_detections[0];
  int box_cnt = 0;
  OD_Result od_result;
  Box box;
  for (int i=0; i<num_detections; i++) {
    if (od_result_raw.scores[i] >= this->confidence_score_threshold) {
      box.xmin = (int)round((od_result_raw.boxes[i*4+1] * img_width));
      box.ymin = (int)round((od_result_raw.boxes[i*4+0] * img_height));
      box.xmax = (int)round((od_result_raw.boxes[i*4+3] * img_width));
      box.ymax = (int)round((od_result_raw.boxes[i*4+2] * img_height));

      if (this->visible) {
        cvRectangle(img, cvPoint(box.xmin, box.ymin), cvPoint(box.xmax, box.ymax), CV_RGB(0, 255, 255));
      }
      if (this->verbose) {
        int idx = od_result_raw.label_ids[i]+1;
        std::string label_name = this->label_id_to_name[idx].as<std::string>().c_str();
        std::cout<<"Box_"<<box_cnt<<"("<<od_result_raw.scores[i]<<", "<<label_name<<"): [" \
                <<box.xmin<<", "<<box.ymin<<", "<<box.xmax<<", "<<box.ymax<<"]"<<std::endl;
      }
      box_cnt++;
      od_result.boxes.push_back(box);
      od_result.scores.push_back(od_result_raw.scores[i]);
      od_result.label_ids.push_back(od_result_raw.label_ids[i]);
    }
  }
  od_result.num_detections = box_cnt;
  if (this->verbose) {
    std::cout<<"Total box number: "<<box_cnt<<std::endl;
  }
  if (this->visible) {
    cvShowImage("Drawing Graphics", img);
    cvWaitKey(0);
  }
  return od_result;
}

void ObjectDetection::add_annotation(const char* img_path, OD_Result od_result) {
  Json::Value annotation;
  std::string img_path_str = std::string(img_path);
  std::string filename = img_path_str.substr(img_path_str.find_last_of("/\\") + 1);
  int file_size = get_image_size(img_path);
  std::string annotation_id = filename+std::to_string(file_size);
  annotation["fileref"] = "";
  annotation["base64_img_data"] = "";
  annotation["size"] = file_size;
  annotation["filename"] = filename.c_str();
  Json::Value regions = raw_to_regions(od_result);
  annotation["regions"] = regions;
  this->annotations[annotation_id] = annotation;
}

Json::Value ObjectDetection::raw_to_regions(OD_Result od_result) {
  std::string label_name;
  Json::Value regions = Json::Value();
  Json::Value region;
  Json::Value shape_attributes; 
  Json::Value region_attributes; 
  Json::Value sub_region_attributes; 
  std::string box_cnt_str;
  std::string label_id_str; 
  for (int i=0; i<od_result.num_detections; i++) {
    shape_attributes["name"] = "rect";
    shape_attributes["x"] = od_result.boxes[i].xmin;
    shape_attributes["y"] = od_result.boxes[i].ymin;
    shape_attributes["width"] = od_result.boxes[i].xmax - od_result.boxes[i].xmin;
    shape_attributes["height"] = od_result.boxes[i].ymax - od_result.boxes[i].ymin;
    int idx = od_result.label_ids[i]+1;
    std::string label_name = this->label_id_to_name[idx].as<std::string>();
    sub_region_attributes["0"] = label_name.c_str();
    region_attributes[this->attribute] = sub_region_attributes;
    region["shape_attributes"] = shape_attributes;
    region["region_attributes"] = region_attributes;
    box_cnt_str = std::to_string(i);
    regions[box_cnt_str.c_str()] = region;
  }
  return regions;
}

void ObjectDetection::write_to_file() {
  Json::StyledWriter writer; 
  std::string output_anno = writer.write(this->annotations); 
  const char* buffer = output_anno.c_str();
  int len = output_anno.length();
  FILE* fp = fopen(this->inference_result_file.c_str(), "wb");
  if(fp == nullptr) {
    exit(1);
  } 
  size_t fileSize = fwrite(buffer, 1, len, fp);
  fclose(fp);
}

double ObjectDetection::get_first_sess_run_elapsed_time() {
  return this->first_sess_run_elapsed_time.count();
}

double ObjectDetection::get_average_sess_run_elapsed_time() {
  double average_sess_run_elapsed_time = \
    (this->total_sess_run_elapsed_time.count() / this->inference_counter);
  return average_sess_run_elapsed_time;
}

int ObjectDetection::get_inference_counter() {
  return this->inference_counter;
}

void ObjectDetection::reset_sess_run_elapsed_time() {
  this->first_sess_run_elapsed_time = std::chrono::seconds(0);
  this->total_sess_run_elapsed_time = std::chrono::seconds(0);
  this->inference_counter = 0;
}

long ObjectDetection::get_model_size() {
  return this->model_size;
}

int ObjectDetection::get_image_size(const char* img_path) // path to file
{
    FILE *p_file = NULL;
    p_file = fopen(img_path, "rb");
    fseek(p_file,0,SEEK_END);
    int size = ftell(p_file);
    fclose(p_file);
    return size;
}

void ObjectDetection::process_mem_usage(double& vm_usage, double& resident_set) {
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0.0;
   resident_set = 0.0;

   // 'file' stat seems to give the most reliable results
   //
   ifstream stat_stream("/proc/self/stat",ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   //
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
   //
   unsigned long vsize;
   long rss;
   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   stat_stream.close();
   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages

   vm_usage     = vsize / 1024.0;
   resident_set = rss * page_size_kb;
}

void ObjectDetection::close() {
  std::cout<<"close() "<<std::endl;
}

