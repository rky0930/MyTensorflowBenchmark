#include "yaml-cpp/yaml.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// our data types
int main()
{
   std::ifstream fin("monsters.yaml");
  YAML::Node doc = YAML::Load(fin);

   for(unsigned i=0;i<doc.size();i++) {
      YAML::Node node = doc[i]["powers"];
      std::cout << node[1]["name"]<< "\n";
   }

   return 0;
}