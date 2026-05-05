#include "tests.cuh"
#include "cpu_reference.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include "kernels.cuh"

//This .cu file is where the tests are performed
//Again it's Naive VS Shared Mem VS SHMem + Coarsening VS cuBLAS

//runTEst runs all of my kernels and outputs the GFLOPS and speedups
Result runTest(int N, int batchCount) {
    Result result;
    result.naiveGFLOPS = 0;
    result.sharedGFLOPS = 0;
    result.cuBLASGFLOPS = 0;
    result.speedup = 0;
    result.coarsenedGFLOPS = 0.0;
    result.sharedSpeedup = 0.0;
    
    //number of total elements throughout all the matrices
    int totalSize = batchCount * N * N;
    
    //std::cout << "\nMatrix Size: " << N << "x" << N <<std::endl;
    //std::cout << "\nBatch Count: " << batchCount << std::endl;

    float* A = new float[totalSize];
    float* B = new float[totalSize];
    float* C = new float[totalSize];
    

    //Fill A and B with simple test values

    for (int i = 0; i < totalSize; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    //CPU Baseline
    CPUBatchedGemm(A, B, C, batchCount, N);

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc(&d_A, totalSize * sizeof(float));
    cudaMalloc(&d_B, totalSize * sizeof(float));
    cudaMalloc(&d_C, totalSize * sizeof(float));

    cudaMemcpy(d_A, A, totalSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, totalSize * sizeof(float), cudaMemcpyHostToDevice);

    //NAIVE TESTING :D
    //float milliseconds = runNaiveLargeCuda(d_A, d_B, d_C, batchCount, N);

    //To help compute GFLOPS
    double flops = 2.0 * batchCount * N * N * N;

    /*double naiveGFLOPS = 0.0;
    double sharedGFLOPS = 0.0;
    double cuBLASGFLOPS = 0.0;
    double speedup = 0.0;*/

    if (N <= 32) {
        
        cudaMemset(d_C, 0, totalSize * sizeof(float));
        float naiveMs = runNaiveCuda(d_A, d_B, d_C, batchCount, N);
        
        cudaMemset(d_C, 0, totalSize * sizeof(float));
        float sharedMs = runSharedCuda(d_A, d_B, d_C, batchCount, N);

        result.naiveGFLOPS = flops / (naiveMs / 1000.0) / 1e9;
        result.sharedGFLOPS = flops / (sharedMs / 1000.0) / 1e9;
        result.sharedSpeedup = naiveMs / sharedMs;
        
        //Optimized
        if (N == 32) {
            cudaMemset(d_C, 0, totalSize * sizeof(float));
            float coarsenedMs = runCoarsenedCuda(d_A, d_B, d_C, batchCount, N);

            result.coarsenedGFLOPS = flops / (coarsenedMs / 1000.0) / 1e9;
            result.speedup = naiveMs / coarsenedMs;
        }

        else {
            result.coarsenedGFLOPS = 0.0;
            result.speedup = 0.0;
        }

        //std::cout << "Naive CUDA time: " << naiveMs << " ms" << std::endl;
        //std::cout << "Naive CUDA GFLOPS: " << naiveGFLOPS << std::endl;

        //std::cout << "Shared CUDA time: " << sharedMs << " ms" << std::endl;
        //std::cout << "Shared CUDA GFLOPS: " << sharedGFLOPS << std::endl;

        //std::cout << "Speedup: " << (naiveMs / sharedMs) << "x" << std::endl;
    }

    else {
        cudaMemset(d_C, 0, totalSize * sizeof(float));
        //To test and see larger matrices
        float naiveMs = runNaiveLargeCuda(d_A, d_B, d_C, batchCount, N);

        result.naiveGFLOPS = flops / (naiveMs / 1000.0) / 1e9;

        //std::cout << "Naive Large CUDA time: " << naiveMs << " ms" << std::endl;
        //std::cout << "Naive Large CUDA GFLOPS: " << naiveGFLOPS << std::endl;
    }

    //cuBLAS Part
    cudaMemset(d_C, 0, totalSize * sizeof(float));
    float cuBLASMs = runCuBLAS(d_A, d_B, d_C, batchCount, N);

    result.cuBLASGFLOPS = flops / (cuBLASMs / 1000.0) / 1e9;

    //std::cout << "cuBLAS StridedBatched time: " << cuBLASMs << " ms" << std::endl;
    //std::cout << "cuBLAS StridedBatched GFLOPS: " << cuBLASGFLOPS << std::endl;

    //cudaMemcpy(C_gpu, d_C, totalSize * sizeof(float), cudaMemcpyDeviceToHost);


    //Correctness for CPU
    /*bool correct = true;

    for (int i = 0; i < totalSize; i++) {
        if (fabs(C[i] - C_gpu[i]) > 1e-4) {
            correct = false;
        }
    }

    if (correct) {
        std::cout << "GPU matches CPU!" << std::endl;
    }

    else {
        std::cout << "GPU does NOT match CPU." << std::endl;
    }
    //Print Results
    //std::cout << "CPU Result:" << std::endl;
    std::cout << "C[0] = " << C[0] << std::endl;
    std::cout << "C_gpu[0] = " << C_gpu[0] << std::endl;*/

    

    //delete[]C_gpu;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[]A;
    delete[]B;
    delete[]C;

    return result;
}

//To run and time Naive kernel
float runNaiveCuda(float* d_A, float*d_B, float* d_C, int batchCount, int N) {
    dim3 block(N, N);
    dim3 grid(1, 1, batchCount);

    //Warmup
    batchedGemmNaive<<<grid, block>>>(d_A, d_B, d_C, batchCount, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    batchedGemmNaive<<<grid, block>>>(d_A, d_B, d_C, batchCount, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;


}

//Run and time shmem kernel
float runSharedCuda(float* d_A, float*d_B, float* d_C, int batchCount, int N) {
    dim3 block(N, N);
    dim3 grid(1, 1, batchCount);

    batchedGemmShared<<<grid, block>>>(d_A, d_B, d_C, batchCount, N);
    cudaDeviceSynchronize();

    cudaEvent_t start2, stop2;

    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);

    cudaEventRecord(start2);

    batchedGemmShared<<<grid, block>>>(d_A, d_B, d_C, batchCount, N);
    
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    float sharedMemMilliseconds = 0.0f;
    cudaEventElapsedTime(&sharedMemMilliseconds, start2, stop2);

    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);

    return sharedMemMilliseconds;
}

//To run and time large naive
float runNaiveLargeCuda(float* d_A, float*d_B, float* d_C, int batchCount, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16, batchCount);

    //Warmup
    
    batchedGemmNaiveLarge<<<grid, block>>>(d_A, d_B, d_C, batchCount, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    batchedGemmNaiveLarge<<<grid, block>>>(d_A, d_B, d_C, batchCount, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;

}

//To run and time optimized kernel (shmem + coarse)
float runCoarsenedCuda(float* d_A, float* d_B, float* d_C, int batchCount, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 31) / 32, (N + 31) / 32, batchCount);

    //Warmup
    
    batchedGemmSharedCoarsened<<<grid, block>>>(d_A, d_B, d_C, batchCount, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    batchedGemmSharedCoarsened<<<grid, block>>>(d_A, d_B, d_C, batchCount, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

//to run and time cuBLASS strided batched GEMM
float runCuBLAS(float* d_A, float*d_B, float* d_C, int batchCount, int N) {

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    long long stride = N * N;

    cudaEvent_t start3;
    cudaEvent_t stop3;
    cudaEventCreate(&start3);
    cudaEventCreate(&stop3);

    //Warmup
    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, stride, d_A, N,stride,  &beta, d_C, N, stride, batchCount);        
        
    cudaDeviceSynchronize();

    cudaEventRecord(start3);

    cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, stride, d_A, N,stride,  &beta, d_C, N, stride, batchCount);        

    cudaEventRecord(stop3);
    cudaEventSynchronize(stop3);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start3, stop3);

    cudaEventDestroy(start3);
    cudaEventDestroy(stop3);

    cublasDestroy(handle);

    return ms;
}
