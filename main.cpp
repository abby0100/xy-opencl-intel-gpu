#include <iostream>
#include <vector>
#include <CL/cl.h>

using namespace std;

#define check_error(status, msg)	\
	if(status != CL_SUCCESS) {		\
		fprintf(stderr, "%s, error happened at\t%s: %d\n", msg, __FILE__, __LINE__);	\
		exit(1);					\
	}

void callback(const char* errInfo, const void*, size_t, void*) {
	cout << "Context callback errInfo:\t" << errInfo << endl;
}

void loadProgramSource(const char** files, size_t number, char** buffer, size_t* sizes) {
	cout << "loadProgramSource" << endl;
	for(size_t i=0; i<number; ++i) {
		FILE* file = fopen(files[i], "r");
		if(!file) {
			cerr << "Failed to open OpenCL kernel file:\t" << files[i] << endl;
			exit(1);
		}

		fseek(file, 0, SEEK_END);
		sizes[i] = ftell(file);
		rewind(file);

		buffer[i] = new char[sizes[i] + 1];
		fread(buffer[i], sizeof(char), sizes[i], file);
		buffer[i][sizes[i]] = 0;

		fclose(file);
	}
}

void displayPlatformInfo(cl_platform_id id, cl_platform_info param_name, const char* paramNameAsStr) {

    cl_int error = 0;
    size_t paramSize = 0;

    error = clGetPlatformInfo(id, param_name, 0, NULL, &paramSize);
    char* moreInfo = (char*) alloca(sizeof(char) * paramSize);

    error = clGetPlatformInfo(id, param_name, paramSize, moreInfo, NULL);
    if(error != CL_SUCCESS) {
        perror("Unable to find any OpenCL platform information, exiting...");
        exit(1);
    }
    printf("%s:\t%s\n", paramNameAsStr, moreInfo);
}

void displayDeviceDetails(cl_device_id id, cl_device_info param_name, const char* paramNameAsStr) {
	cl_int error = 0;
	size_t paramSize = 0;

	error = clGetDeviceInfo(id, param_name, 0, NULL, &paramSize);
	if(error != CL_SUCCESS) {
		perror("Unable to obtain device info for param, exiting...");
		exit(1);
	}

	switch(param_name) {
		case CL_DEVICE_TYPE: {
			cl_device_type* dev_type = (cl_device_type*) alloca (sizeof(cl_device_type) * paramSize);

			error = clGetDeviceInfo(id, param_name, paramSize, dev_type, NULL);
			if(error != CL_SUCCESS) {
				perror("Unable to obtain device info for dev_type, exiting...");
				exit(1);
			}

			switch(*dev_type) {
			case CL_DEVICE_TYPE_CPU:
				printf("CPU detected\n");
				break;
			case CL_DEVICE_TYPE_GPU:
				printf("GPU detected\n");
				break;
			case CL_DEVICE_TYPE_DEFAULT:
				printf("default device detected\n");
				break;
			}
		}
		break;
		case CL_DEVICE_VENDOR_ID:
		case CL_DEVICE_MAX_COMPUTE_UNITS:
		case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: {
			cl_uint* ret = (cl_uint*) alloca (sizeof(cl_uint) * paramSize);
			error = clGetDeviceInfo(id, param_name, paramSize, ret, NULL);
			if(error != CL_SUCCESS) {
				perror("Unable to obtain device info for param, exiting...");
				exit(1);
			}

			switch(param_name) {
			case CL_DEVICE_VENDOR_ID:
				printf("\tVENDOR ID:\t\t\t\t0x%x\n", *ret);
				break;
			case CL_DEVICE_MAX_COMPUTE_UNITS:
				printf("\tMaximum number of parallel compute units:\t%d\n", *ret);
				break;
			case CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:
				printf("\tMaximu dimensions for global/local work-item IDs:\t%d\n", *ret);
				break;
			}
		}
		break;
		case CL_DEVICE_MAX_WORK_ITEM_SIZES: {
			cl_uint maxWIDimensions;
			size_t* ret = (size_t*) alloca (sizeof(size_t) * paramSize);
			error = clGetDeviceInfo(id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &maxWIDimensions, NULL);
			if(error != CL_SUCCESS) {
				perror("Unable to obtain device info for param, exiting...");
				exit(1);
			}
			printf("\tMaximum number of work-items in each dimension: ( ");
			for (cl_int i=0; i<maxWIDimensions; ++i) {
				printf("%d ", (int)ret[i]);
			}
			printf(" )\n");
		}
		break;
		case CL_DEVICE_MAX_WORK_GROUP_SIZE: {
			size_t* ret = (size_t*) alloca (sizeof(size_t) * paramSize);
			error = clGetDeviceInfo(id, param_name, paramSize, ret, NULL);
			if(error != CL_SUCCESS) {
				perror("Unable to obtain device info for param, exiting...");
				exit(1);
			}
			printf("\tMaximun number of work-items in a work-group:\t%d\n", (int)*ret);
		}
		break;
		case CL_DEVICE_GLOBAL_MEM_SIZE: {
			size_t* ret = (size_t*) alloca (sizeof(size_t) * paramSize);
			error = clGetDeviceInfo(id, param_name, paramSize, ret, NULL);
			if(error != CL_SUCCESS) {
				perror("Unable to obtain device info for param, exiting...");
				exit(1);
			}
			printf("\tGlobal memory of Device (MB):\t\t\t%d\n", (int)(*ret/1024/1024));
		}
		break;
		case CL_DEVICE_EXTENSIONS: {
			char extensionInfo[4096];
			error = clGetDeviceInfo(id, param_name, paramSize, extensionInfo, NULL);
			printf("\tSupported extensions: %s\n", extensionInfo);
		}
		break;
	}
}

void displayDeviceInfo(cl_platform_id id, cl_device_type dev_type) {
    cl_int error = 0;
    cl_uint numOfDevices = 0;

    error = clGetDeviceIDs(id, dev_type, 0, NULL, &numOfDevices);
    if(error != CL_SUCCESS) {
        perror("Unable to find any OpenCL compliant devices, exiting...");
        exit(1);
    }

    cl_device_id* devices = (cl_device_id*) alloca(sizeof(cl_device_id) * numOfDevices);
    error = clGetDeviceIDs(id, dev_type, numOfDevices, devices, NULL);
    if(error != CL_SUCCESS) {
        perror("Unable to find any OpenCL compliant devices info, exiting...");
        exit(1);
    }

    printf("Number of detected OpenCL devices:\t\t%d\n", numOfDevices);
    for(cl_uint i=0; i<numOfDevices; ++i) {
        displayDeviceDetails(devices[i], CL_DEVICE_TYPE,        "CL_DEVICE_TYPE");
        displayDeviceDetails(devices[i], CL_DEVICE_VENDOR_ID,   "CL_DEVICE_VENDOR_ID");
        displayDeviceDetails(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS,   "CL_DEVICE_MAX_COMPUTE_UNITS");
        displayDeviceDetails(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,    "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
        displayDeviceDetails(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
        displayDeviceDetails(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
        displayDeviceDetails(devices[i], CL_DEVICE_EXTENSIONS,  "CL_DEVICE_EXTENSIONS");
    }
}

void clinfo() {

	// platforms
	cl_int status = CL_SUCCESS;
	cl_uint numPlatforms = 0;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	cout << "Number of Platforms:\t" << numPlatforms << endl;

	std::vector<cl_platform_id> platforms;
	platforms.resize(numPlatforms);
	status |= clGetPlatformIDs(numPlatforms, platforms.data(), NULL);

	//for(cl_uint i=0; i<numPlatforms; ++i) {
	//	cout << "###############################################################" << endl;
	//	displayPlatformInfo(platforms[i], CL_PLATFORM_VERSION,	"CL_PLATFORM_VERSION");
	//	displayPlatformInfo(platforms[i], CL_PLATFORM_VENDOR,	"CL_PLATFORM_VENDOR");
	//	displayPlatformInfo(platforms[i], CL_PLATFORM_NAME,		"CL_PLATFORM_NAME");
	//	displayPlatformInfo(platforms[i], CL_PLATFORM_PROFILE,	"CL_PLATFORM_PROFILE");
	//	displayPlatformInfo(platforms[i], CL_PLATFORM_EXTENSIONS,	"CL_PLATFORM_EXTENSIONS");

	//	cout << endl;
	//	displayDeviceInfo(platforms[i], CL_DEVICE_TYPE_ALL);
	//}

	// devices: 1 - Intel Iris Plus 655
	cl_platform_id platform = platforms[1];
	cl_uint numDevices = 0;
	status |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
	cout << "Number of Devices:\t" << numDevices << endl;
	
	std::vector<cl_device_id> devices;
	devices.resize(numDevices);
	status |= clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
	check_error(status, "Failed to get devices.");

	// context
	cl_context context = clCreateContext(NULL, numDevices, devices.data(), &callback, NULL, &status);
	check_error(status, "Failed to create context.");
	cout << "context created." << endl;

	// queue
	cl_command_queue queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
	check_error(status, "Failed to create queue.");
	cout << "queue created." << endl;

	// program
	cl_int numCL = 1;
	size_t kernel_sizes[numCL];
	char* kernel_buffers[numCL];
	const char* kernel_files[numCL] = {"compute_distance.cl"};
	loadProgramSource(kernel_files, numCL, kernel_buffers, kernel_sizes);

	cl_program program = clCreateProgramWithSource(context, numCL, (const char**)kernel_buffers, kernel_sizes, &status);
	check_error(status, "Failed to create program.");

	// build
	const char options[] = "-cl-finite-math-only -cl-no-signed-zeros";
	status |= clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if(status != CL_SUCCESS) {
		size_t log_size = 0;
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char program_log[log_size + 1];
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size+1, program_log, NULL);
		program_log[log_size] = 0;
		cout << "\n=== Error === \n\n" << program_log << "\n=============" << endl;
		exit(1);
	}
	cout << "program created." << endl;

	// kernel
	cl_kernel kernel = clCreateKernel(program, "compute_distance", &status);
	check_error(status, "Failed to create kernel");
	cout << "kernel created." << endl;
	
	// buffer

	// kernel args

	// enqueue
	cl_int dim = 2;
	size_t gsizes[] = {4,4};
	size_t lsizes[] = {2,2};
	status |= clEnqueueNDRangeKernel(queue, kernel, dim, NULL, gsizes, lsizes, 0, NULL, NULL);
	check_error(status, "Failed to enqueue NDRange kernel.");

	// finish
	status |= clFinish(queue);
	// release
	if(kernel) {
		clReleaseKernel(kernel);  
	}
	if(program) {
		clReleaseProgram(program);
	}
	if(queue) {
		clReleaseCommandQueue(queue);
	}
	if(context) {
		clReleaseContext(context);
	}

}

int main(int argc, char** argv) {

	clinfo();
	return 0;
}
