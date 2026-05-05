#include "kernels.cuh"
#define MAX_N 32

//My Naive
//Slowest
//Only computes asingle output for C
__global__
void batchedGemmNaive(float* A, float* B, float* C, int batchCount, int N) {

    int batch = blockIdx.z;
    int row = threadIdx.y;
    int col = threadIdx.x;

    if (batch >= batchCount || row >= N || col >= N) {
        return ;
    }

    float* Ab = A + batch * N * N;
    float* Bb = B + batch * N * N;
    float* Cb = C + batch * N * N;

    float sum = 0.0f;
    //Loading from Global Mem
    for (int k = 0; k < N; k++) {
        sum += Ab[row * N + k] * Bb[k * N + col];
    }

    Cb[row * N + col] = sum;



}

//Batched GEMM w/ Shmem
//Only one output is still made by each thread
//But A and B are placed into Shared Mem for less usage of Global Mem
__global__ 
void batchedGemmShared(float* A, float* B, float* C, int batchCount, int N) {
    __shared__ float As[MAX_N][MAX_N];
    __shared__ float Bs[MAX_N][MAX_N];

    int batch = blockIdx.z;
    int row = threadIdx.y;
    int col = threadIdx.x;

    float* Ab = A + batch * N * N;
    float* Bb = B + batch * N * N;
    float* Cb = C + batch * N * N;

    As[row][col] = Ab[row * N + col];
    Bs[row][col] = Bb[row * N + col];

    __syncthreads();

    float sum = 0.0f;

    for (int k = 0; k < N; k++) {
        sum += As[row][k] * Bs[k][col];
    }

    Cb[row * N + col] = sum;

}

//Kernel for Naive
//for large matrices
//One output still conputed by each thread
//Written into project to compare against cuBLAS batched GEMM strided
//and show how B GEMM Strided is much more efficient for way bigger matrices
__global__
void batchedGemmNaiveLarge(float* A, float* B, float* C, int batchCount, int N) {

    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchCount || row >= N || col>= N) {
        return;
    }

    float* Ab = A + batch * N * N;
    float* Bb = B + batch * N * N;
    float* Cb = C + batch * N * N;

    float sum = 0.0f;

    for (int k = 0; k < N; k++) {
        sum += Ab[row * N + k] * Bb[k * N + col];
    }

    Cb[row * N + col] = sum;


}

//Best/Optimized kernel for project
//This uses shared memory and Coarsening
//Each thread computes 4 outputs on a 2 by 2 block
//CF, coursening factor I used was 4
__global__ void batchedGemmSharedCoarsened(float* A, float* B, float* C, int batchCount, int N) {
    
    __shared__ float As[MAX_N][MAX_N];
    __shared__ float Bs[MAX_N][MAX_N];

    int batch = blockIdx.z;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = ty;
    int col = tx;

    int row2 = row + 16;
    int col2 = col + 16;

    float* Ab = A + batch * N * N;
    float* Bb = B + batch * N * N;
    float* Cb = C + batch * N * N;

    //4 elements placed in SHMEM
    As[row][col] = Ab[row * N + col];
    As[row][col2] = Ab[row * N + col2];
    As[row2][col] = Ab[row2 * N + col];
    As[row2][col2] = Ab[row2 * N + col2];

    Bs[row][col] = Bb[row * N + col];
    Bs[row][col2] = Bb[row * N + col2];
    Bs[row2][col] = Bb[row2 * N + col];
    Bs[row2][col2] = Bb[row2 * N + col2];

    __syncthreads();
    //OUtputs
    float c00 = 0.0f;
    float c01 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;

    #pragma unroll
    for (int k = 0; k < 32; k++) {
        c00 += As[row][k] * Bs[k][col];
        c01 += As[row][k] * Bs[k][col2];
        c10 += As[row2][k] * Bs[k][col];
        c11 += As[row2][k] * Bs[k][col2];

    }

    Cb[row * N + col] = c00;
    Cb[row * N + col2] = c01;
    Cb[row2 * N + col] = c10;
    Cb[row2 * N + col2] = c11;


}