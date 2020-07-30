#include <iostream>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <CL/cl.h>

using namespace std;
//const int width = 4;
//const int width = 8;
//const int width = 16;
const int width = 64;

#define check_error(status, msg)	\
	if(status != CL_SUCCESS) {		\
		fprintf(stderr, "%s, error happened at\t%s: %d\n", msg, __FILE__, __LINE__);	\
		exit(1);					\
	}

void callback(const char* errInfo, const void*, size_t, void*) {
	cout << "Context callback errInfo:\t" << errInfo << endl;
}

cl_ulong getStartEndTime(cl_event event) {
	cl_int status;
	
	cl_ulong start, end;
	status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
	check_error(status, "Failed to query event start time");
	status = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
	check_error(status, "Failed to query event end time");
	
	return end - start;
}

cl_ulong getStartEndTime(cl_event *events, unsigned num_events) {
	cl_int status;
	
	cl_ulong min_start = 0;
	cl_ulong max_end = 0;
	for(unsigned i = 0; i < num_events; ++i) {
		cl_ulong start, end;
		status = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
		check_error(status, "Failed to query event start time");
		status = clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
		check_error(status, "Failed to query event end time");
		
		if(i == 0) {
			min_start = start;
			max_end = end;
		}
		else {
			if(start < min_start) {
				min_start = start;
			}
			if(end > max_end) {
				max_end = end;
			}
		}
	}
	
	return max_end - min_start;
}

template <typename T>
void validateBuffer(T* buffer, size_t buffsize) {
	std::cout << "\nvalidateBuffer:" << std::endl;

	for(size_t i=0; i<buffsize; ++i) {
		if(i % width == 0)
			std::cout << std::endl;
		std::cout << buffer[i] << " ";
	}
	std::cout << std::endl;
}
template <typename T>
void checkResults(T* ha, T* hb, size_t rowA, size_t colA, size_t colB) {
	std::cout << "\ncheckResults:" << std::endl;
	for(size_t i=0; i<rowA; ++i) {
		for(size_t j=0; j<colB; ++j) {
			float sum = 0.0f;
			for(size_t k=0; k<colA; ++k) {
				sum += ha[i*colA + k] * hb[k*colB + j];
			}
			std::cout << sum << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}
template <typename T>
void checkDistance(T* ha, T* hb, size_t dim, size_t number) {
	std::cout << "\ncheckDistance:" << std::endl;
	struct timeval start;
	struct timeval end;

	gettimeofday(&start, NULL);
	for(size_t j=0; j<number; ++j) {
		//if(j % width == 0)
		//	std::cout << std::endl;
		float sum = 0.0f;
		for(size_t i=0; i<dim; ++i) {
			sum += ha[j*dim + i] * hb[j*dim + i];
		}
		//std::cout << sum << " ";
	}
	//std::cout << std::endl;

	gettimeofday(&end, NULL);
	std::cout << "Reference Kernel time:\t" << ((end.tv_sec - start.tv_sec)*1e6 + (end.tv_usec - start.tv_usec)) << " us" << std::endl;
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

void clinfo(int deviceid) {
	cout << "clinfo deviceid:\t" << deviceid << endl;

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
	cl_platform_id platform = platforms[0];
	//cl_platform_id platform = platforms[1];
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
	//const char* kernel_files[numCL] = {"compute_distance.cl"};
	const char* kernel_files[numCL] = {"../compute_distance.cl"};
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
	//cl_int vector_dims = 16;
	cl_int vector_dims = 64;
	//cl_int vector_nums = 16;
	//cl_int vector_nums = 1024;
	cl_int vector_nums = 8096;
	cl_int vector_size = vector_dims * vector_nums;

	cl_int colA = sqrt(vector_dims);
	cl_int rowA = sqrt(vector_dims);
	cl_int colB = sqrt(vector_dims);
	float ha[vector_size];
	float hb[vector_size];
	float hc[vector_nums];

	cout << "init host data..." << endl;
	for(cl_int j=0; j<vector_nums; ++j) {
		for(cl_int i=0; i<vector_dims; ++i) {
			//ha[j*vector_dims + i] = (float) (i+1);
			//hb[j*vector_dims + i] = (float) (i+1);
			ha[j*vector_dims + i] = (float) (1);
			hb[j*vector_dims + i] = (float) (1);
		}
		hc[j] = (float) (0);
	}
	cout << "init host data done." << endl;
	//validateBuffer(ha, vector_size);
	//validateBuffer(hb, vector_size);
	validateBuffer(hc, vector_nums);
	//checkResults(ha, hb, rowA, colA, colB);
	checkDistance(ha, hb, vector_dims, vector_nums);

	// kernel args
	cl_mem da = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * vector_size, ha, &status);
	cl_mem db = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * vector_size, hb, &status);
	cl_mem dc = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * vector_nums, NULL, &status);

	status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*) &da);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*) &db);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*) &dc);
	//status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*) NULL);
	status |= clSetKernelArg(kernel, 3, sizeof(float) * 16, (void*) NULL);
	check_error(status, "Failed to set kernel arg 2");

	status |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void*) &colA);
	status |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void*) &colB);
	check_error(status, "Failed to set kernel args.");

	// enqueue
	cl_int dim = 2;
	//size_t gsizes[] = {48,16*16*16};
	//size_t lsizes[] = {16,16};
	//size_t gsizes[] = {8,2*16};
	size_t gsizes[] = {8,2*vector_nums};
	size_t lsizes[] = {8,2};

	cl_event event;
	//status |= clEnqueueNDRangeKernel(queue, kernel, dim, NULL, gsizes, lsizes, 0, NULL, NULL);
	status |= clEnqueueNDRangeKernel(queue, kernel, dim, NULL, gsizes, lsizes, 0, NULL, &event);
	check_error(status, "Failed to enqueue NDRange kernel.");

	// finish
	status |= clFinish(queue);
	//clWaitForEvents(1 , &event);
	cl_ulong duration = getStartEndTime(event);
	cout << (deviceid == 0 ? "CPU" : "GPU") << " Kernel time:\t" << duration/1000 << " us" << endl;

	// read
	//status |= clEnqueueReadBuffer(queue, dc, CL_TRUE, 0, sizeof(float) * vector_dims, hc, 0, NULL, NULL);
	status |= clEnqueueReadBuffer(queue, dc, CL_TRUE, 0, sizeof(float) * vector_nums, hc, 0, NULL, NULL);
	check_error(status, "Failed to read buffer from device.");
	cout << "Read from device buffer." << endl;
	//validateBuffer(hc, vector_dims);
	validateBuffer(hc, vector_nums);

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
	if(argc > 1) {
		clinfo(atoi(argv[1]));
	}
	return 0;
}
