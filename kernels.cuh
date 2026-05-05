#ifndef KERNELS_CUH
#define KERNELS_CUH

__global__ void batchedGemmNaive(float* A, float* B, float* C, int batchCount, int N);
__global__ void batchedGemmShared(float* A, float* B, float* C, int batchCount, int N);
__global__ void batchedGemmNaiveLarge(float* A, float* B, float* C, int batchCount, int N);
__global__ void batchedGemmSharedCoarsened(float* A, float* B, float* C, int batchCount, int N);

#endif
