#ifndef OPENCL
#define OPENCL
#endif

__kernel void compute_distance() {
	int idx = get_global_id(0);
	int idx1 = get_global_id(1);
	printf("[GPU] idx(%d,%d)\n", idx,idx1);
}
