sudo apt install libyaml-cpp-dev
sudo apt install libjsoncpp-dev
ldconfig

c++ -o test_yaml test_yaml.cpp -lyaml-cpp
c++ -o test_json test_json.cpp -ljsoncpp