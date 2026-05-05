#ifndef TEST_CUH
#define TESTS_CUH

struct Result {
    double naiveGFLOPS;
    double sharedGFLOPS;
    double cuBLASGFLOPS;
    double speedup;
    double coarsenedGFLOPS;
    double sharedSpeedup;
};

Result runTest(int N, int batchCount);

float runNaiveCuda(float* d_A, float*d_B, float* d_C, int batchcount, int N);

float runSharedCuda(float* d_A, float*d_B, float* d_C, int batchcount, int N);

float runNaiveLargeCuda(float* d_A, float*d_B, float* d_C, int batchcount, int N);

float runCuBLAS(float* d_A, float*d_B, float* d_C, int batchcount, int N);

float runCoarsenedCuda(float* d_A, float*d_B, float* d_C, int batchcount, int N);

#endif