#ifndef OPENCL
#define OPENCL
#endif

__kernel void compute_distance(__global const float* restrict da, __global const float* restrict db, __global float* restrict dc, 
		__local float* results, int colA, int colB) {
	const int simd = 4;

	int idx = get_global_id(0);
	int idx1 = get_global_id(1);
	int lidx = get_local_id(0);
	int lidx1 = get_local_id(1);

	int gidx = get_group_id(0);
	int gidx1 = get_group_id(1);

	//int index = idx * colA + idx1 * simd;
	int index = lidx * colA + lidx1 * simd;
	int shareid = lidx * 2 + lidx1;

	results[shareid] = 0.0f;
	float tmp = 0.0f;
	barrier(CLK_LOCAL_MEM_FENCE);

	//printf("[GPU] idx(%d,%d) grp(%d,%d) col(%d,%d)\n", idx,idx1, gidx,gidx1, colA, colB);
	
	for(int i=0; i<simd; ++i) {
		//results[idx * 2 + idx1] += da[index + i] * db[index + i];
		//results[lidx * 2 + lidx1] += da[index + i] * db[index + i];
		//results[shareid] += da[index + i] * db[index + i];
		tmp = da[index + i] - db[index + i];
		results[shareid] += tmp * tmp; 
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0.0f;
	if(lidx == 0 && lidx1 == 0) {
		//for(int j=0; j<128/simd; ++j) {
		for(int j=0; j<64/simd; ++j) {
			sum += results[j];
		}

		dc[gidx1] = sum;
		//printf("[GPU] dc(%f)\n", dc[gidx1]);
	}
	sum = 0.0f;
}
