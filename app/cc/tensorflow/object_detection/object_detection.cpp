#include "object_detection.hpp"

void free_buffer(void* data, size_t length) { free(data); }

TF_Tensor* create_tensor(TF_DataType data_type,
                        const std::int64_t* dims, std::size_t num_dims,
                        const void* data, std::size_t len) {
  if (dims == nullptr || data == nullptr) {
    return nullptr;
  }
  TF_Tensor* tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
  if (tensor == nullptr) {
    return nullptr;
  }
  std::memcpy(TF_TensorData(tensor), data, std::min(len, TF_TensorByteSize(tensor)));
  return tensor;
}


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
  this->graph_def = this->read_graph(this->frozen_graph_path);
  this->graph = TF_NewGraph();
  this->graph_status = TF_NewStatus();
  this->graph_opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(this->graph, this->graph_def, 
                         this->graph_opts, this->graph_status);
  if (TF_GetCode(this->graph_status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(this->graph_status));
    exit(1);
  } else {
    fprintf(stdout, "Successfully imported graph\n");
  }

  // Create Session 
  this->sess_opts = TF_NewSessionOptions();
  this->sess_status = TF_NewStatus();
  this->sess = TF_NewSession(this->graph, this->sess_opts, this->sess_status);
  if(TF_GetCode(this->sess_status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to create session %s", TF_Message(this->sess_status));
  }
  // Set up input op
  this->input_op = TF_GraphOperationByName(graph, "image_tensor");
  this->input_opout = {input_op, 0};
  this->input_ops.push_back(input_opout);
  // Set up output ops
  this->boxes_op = TF_GraphOperationByName(graph, "detection_boxes");
  this->scores_op = TF_GraphOperationByName(graph, "detection_scores");
  this->classes_op = TF_GraphOperationByName(graph, "detection_classes");
  this->num_detections_op = TF_GraphOperationByName(graph, "num_detections");
  this->boxes_opout = {boxes_op, 0};
  this->scores_opout = {scores_op, 0};
  this->classes_opout = {classes_op, 0};
  this->num_detections_opout = {num_detections_op, 0};
  this->output_ops.push_back(boxes_opout);
  this->output_ops.push_back(scores_opout);
  this->output_ops.push_back(classes_opout);
  this->output_ops.push_back(num_detections_opout);
    std::cout<<"asdfhkjahjklfa"<<std::endl;
  if (this->verbose) {
    std::cout << "Input Op Name: "  << TF_OperationName(input_op) << "\n";
    std::cout << "Output Op Name: " << TF_OperationName(boxes_op) << "\n";
    std::cout << "Output Op Name: " << TF_OperationName(scores_op) << "\n";
    std::cout << "Output Op Name: " << TF_OperationName(classes_op) << "\n";
    std::cout << "Output Op Name: " << TF_OperationName(num_detections_op) << "\n";
  }
}

void ObjectDetection::preprocessing(IplImage* src, IplImage* dst) {
  cvCvtColor(src, dst, CV_BGR2RGB);
}

OD_Result_raw ObjectDetection::sess_run(IplImage* img) {
  ResetOutputValues();
  // Create input variable
  int img_width = img->width;
  int img_height = img->height;
  int img_channel = img->nChannels;  
  const std::vector<std::int64_t> input_dims = {1, img_height, img_width, img_channel};
  int image_size_by_dims = img_height*img_width*img_channel;
  int image_tensor_size = std::min(image_size_by_dims, img->imageSize);
  if (this->verbose) {
    std::cout<<"image_tensor_size: "<<image_tensor_size<<std::endl;
  }
  TF_Tensor* input_value = create_tensor(TF_UINT8,
                                        input_dims.data(), input_dims.size(),
                                        img->imageData, image_tensor_size);
  // TF_Tensor* input_values[1] = {input_value};
  input_values.emplace_back(input_value);
  // Create output variable
  const std::vector<std::int64_t> box_dims = {1, this->max_detections, 4};
  const std::vector<std::int64_t> scores_dims = {1, this->max_detections};
  const std::vector<std::int64_t> classes_dims = {1, this->max_detections};
  const std::vector<std::int64_t> num_detections_dims = {1, 1};
  TF_Tensor* boxes_value = TF_AllocateTensor(TF_FLOAT, box_dims.data(), box_dims.size(), sizeof(float) * 4 * this->max_detections);
  TF_Tensor* scores_value = TF_AllocateTensor(TF_FLOAT, scores_dims.data(), scores_dims.size(), sizeof(float) * this->max_detections);
  TF_Tensor* classes_value = TF_AllocateTensor(TF_FLOAT, classes_dims.data(), classes_dims.size(), sizeof(float) * this->max_detections);
  TF_Tensor* num_detections_value = TF_AllocateTensor(TF_FLOAT, num_detections_dims.data(), num_detections_dims.size(), sizeof(float));
  // TF_Tensor* output_values[4] = {boxes_value, scores_value, classes_value, num_detections_value};
  output_values.emplace_back(boxes_value);
  output_values.emplace_back(scores_value);
  output_values.emplace_back(classes_value);
  output_values.emplace_back(num_detections_value);
  if (this->verbose) {
    std::cout << "Input op info: " << TF_OperationNumInputs(input_op) << "\n";
    std::cout << "Input dims info: (" << TF_Dim(input_value, 0) <<", "<< TF_Dim(input_value, 1) <<", "\
                                      << TF_Dim(input_value, 2) <<", "<< TF_Dim(input_value, 3) <<")"<< "\n";
  }  
  const TF_Output* inputs_ptr = input_ops.empty() ? nullptr : &input_ops[0];
  TF_Tensor* const* input_values_ptr =
      input_values.empty() ? nullptr : &input_values[0];
  const TF_Output* outputs_ptr = output_ops.empty() ? nullptr : &output_ops[0];
  TF_Tensor** output_values_ptr =
      output_values.empty() ? nullptr : &output_values[0];
      
  // Create session
  std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
  start_time = std::chrono::system_clock::now(); 
  TF_SessionRun(this->sess, nullptr,
                inputs_ptr, input_values_ptr, this->input_ops.size(),
                outputs_ptr, output_values_ptr, this->output_ops.size(),
                nullptr, 0, nullptr, this->sess_status);
  end_time = std::chrono::system_clock::now();
  if (this->inference_counter == 0) {
    this->first_sess_run_elapsed_time = (end_time - start_time);
  }else{
    this->total_sess_run_elapsed_time = this->total_sess_run_elapsed_time + (end_time - start_time);
  }
  this->inference_counter++;
  OD_Result_raw od_result_raw;
  od_result_raw.boxes = (float*)TF_TensorData(output_values[0]);
  od_result_raw.scores = (float*)TF_TensorData(output_values[1]);
  od_result_raw.label_ids = (float*)TF_TensorData(output_values[2]);
  od_result_raw.num_detections = (float*)TF_TensorData(output_values[3]);
  TF_DeleteTensor(boxes_value);
  TF_DeleteTensor(scores_value);
  TF_DeleteTensor(classes_value);
  TF_DeleteTensor(num_detections_value);
  DeleteInputValues();
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
        std::cout<<"Box_"<<box_cnt<<"("<<od_result_raw.scores[i]<<", "<<od_result_raw.label_ids[i]<<"): [" \
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

TF_Buffer* ObjectDetection::read_graph(std::string path) {
  const char* file = path.c_str();
  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  //same as rewind(f);

  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);
  this->model_size = fsize;
  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}

void ObjectDetection::DeleteInputValues() {
  for (size_t i = 0; i < input_values.size(); ++i) {
    if (input_values[i] != nullptr) TF_DeleteTensor(input_values[i]);
  }
  input_values.clear();
}

void ObjectDetection::ResetOutputValues() {
  for (size_t i = 0; i < output_values.size(); ++i) {
    if (output_values[i] != nullptr) TF_DeleteTensor(output_values[i]);
  }
  output_values.clear();
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
    sub_region_attributes["0"] = this->label_id_to_name[od_result.label_ids[i]].as<std::string>().c_str();
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
  TF_CloseSession(this->sess, this->sess_status);
  TF_DeleteSession(this->sess, this->sess_status);
  TF_DeleteSessionOptions(this->sess_opts);
  TF_DeleteImportGraphDefOptions(this->graph_opts);
  TF_DeleteGraph(this->graph);
  TF_DeleteStatus(this->graph_status);
}

