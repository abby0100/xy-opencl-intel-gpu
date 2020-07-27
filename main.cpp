#include <iostream>
#include <vector>
#include <CL/cl.h>

using namespace std;

void clinfo() {
	cl_uint numPlatforms = 0;
	clGetPlatformIDs(0, NULL, &numPlatforms);
	cout << "Number of Platforms:\t" << numPlatforms << endl;

	std::vector<cl_platform_id> platforms;
	platforms.resize(numPlatforms);
	
}

int main(int argc, char** argv) {

	clinfo();
	return 0;
}
