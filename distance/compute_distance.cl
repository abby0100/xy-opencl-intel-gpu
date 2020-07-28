#ifndef OPENCL
#define OPENCL
#endif

__kernel void compute_distance(__global const float* restrict da, __global const float* restrict db, __global float* restrict dc, 
		__local float* results, int colA, int colB) {
	const int simd = 4;

	int idx = get_global_id(0);
	int idx1 = get_global_id(1);
	int index = idx * colA + idx1 * simd;

	//printf("[GPU] idx(%d,%d) col(%d,%d)\n", idx,idx1, colA, colB);

	for(int i=0; i<simd; ++i) {
		results[idx * 2 + idx1] += da[index + i] * db[index + i];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0.0f;
	if(idx == 0 && idx1 == 0) {
		for(int j=0; j<128/simd; ++j) {
			sum += results[j];
		}

		*dc = sum;
		printf("[GPU] dc(%f)\n", *dc);
	}
}
