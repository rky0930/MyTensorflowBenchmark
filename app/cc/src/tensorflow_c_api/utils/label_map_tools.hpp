#include "yaml-cpp/yaml.h"
#include <iostream>
#include <fstream>

void load_label_map(std::string label_map_file, std::string& attribute, \
                          YAML::Node& label_id_to_name);
// def load_label_map(label_map_file):
// def convert_label_map(label_id_to_name):
