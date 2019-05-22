#include <iostream> 
#include <fstream>
#include <jsoncpp/json/json.h>

using namespace std;

int main() {
ifstream ifs("profile.json");
Json::Reader reader;
Json::Value obj;
reader.parse(ifs, obj); // Reader can also read strings
cout << "Last name: " << obj["lastname"].asString() << endl;
cout << "First name: " << obj["firstname"].asString() << endl;
return 1;
}