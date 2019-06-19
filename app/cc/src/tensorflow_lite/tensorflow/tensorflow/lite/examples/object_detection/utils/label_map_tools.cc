// import yaml

// def load_label_map(label_map_file):
//     with open(label_map_file, 'r') as stream:
//         try:
//             label_map = yaml.safe_load(stream)
//             attribute = list(label_map.keys())[0]
//             label_id_to_name = label_map[attribute]
//             return attribute, label_id_to_name
//         except yaml.YAMLError as exc:
//             print(exc)

// def convert_label_map(label_id_to_name):
//     if not isinstance(label_id_to_name, dict):
//         raise ValueError("label map is not dictionary type")
//     return dict((_name,_id) for _id, _name in label_id_to_name.items())
#include "label_map_tools.h"

void load_label_map(std::string label_map_file, std::string& attribute, 
                          YAML::Node& label_id_to_name)
{
   std::ifstream fin(label_map_file.c_str());
   YAML::Node doc = YAML::Load(fin);
   for (const auto& kv : doc) {
      attribute = kv.first.as<std::string>(); // prints Foo
      label_id_to_name = kv.second;  // the value
      break;
   }
   // std::cout<<"label_map: ("<<label_map_file<<")"<<std::endl;
   // std::cout<<label_map.first<<std::endl;
}