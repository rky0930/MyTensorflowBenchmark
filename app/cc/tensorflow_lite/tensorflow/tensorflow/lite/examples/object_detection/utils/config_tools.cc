#include "config_tools.h"

YAML::Node load_config(std::string config_file)
{
   std::ifstream fin(config_file.c_str());
   YAML::Node doc = YAML::Load(fin);
   return doc;
}