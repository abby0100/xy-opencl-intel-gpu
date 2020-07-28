#ifndef OPENCL
#define OPENCL
#endif

__kernel void compute_distance(__global const float* restrict da, __global const float* restrict db, __global float* restrict dc, int colA, int colB) {

	int idx = get_global_id(0);
	int idx1 = get_global_id(1);
	int idxa = idx * colA;
	int idxb = idx1;

	//printf("[GPU] idx(%d,%d)\n", idx,idx1);

	float4 lda = (float4) (da[idxa], da[idxa + 1], da[idxa + 2], da[idxa + 3]);
	float4 ldb = (float4) (db[idxb], db[idxb + colB], db[idxb + 2*colB], db[idxb + 3*colB]);
	float4 result = lda*ldb;
	//printf("[GPU] result: %f\n", result.s0 + result.s1 + result.s2 + result.s3);

	int idxc = idx * colB + idx1;
	dc[idxc] = result.s0 + result.s1 + result.s2 + result.s3;
	//printf("[GPU] dc[%d]: %f\n", idxc, dc[idxc]);

	//printf("[GPU] result:%f lda(%f,%f,%f,%f) ldb(%f,%f,%f,%f)\n", result.s0 + result.s1 + result.s2 + result.s3, lda.s0, lda.s1, lda.s2, lda.s3, ldb.s0, ldb.s1, ldb.s2, ldb.s3);
}
