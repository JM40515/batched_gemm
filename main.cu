#include <stdio.h>
#include <iostream>
#include "cpu_reference.h"
#include "kernels.cuh"
#include <cmath>
#include <cublas_v2.h>
#include "tests.cuh"
#include <iomanip>

int main() {
    //Matrix Sizes
    int Ns[] = {8, 16, 32, 64, 128};
    //Batch Sizes
    int batchSizes[] = {128, 1024, 8192};

    int numNs = 5;
    int numBatches = 3;

    //std::cout << "N | Batch | Naive GFLOPS | Shared GFLOPS | cuBLAS GFLOPS | Speedup\n";
    //std::cout << "-------------------------------------------------------------------\n";

    for (int i = 0; i < numNs; i++) {
        
        int N = Ns[i];

        std::cout << "\nMatrix Size: " << N << "x" << N << "\n";

        if (N <= 32) {
            std::cout << "Batch   Naive GFLOPS   Shared GFLOPS   SharedSpeed   Coarsened GFLOPS   Cor Speed   cuBLAS GFLOPS\n";
            std::cout << "-----------------------------------------------------------------------------------------------\n";
        }

        else {
            std::cout << "Batch   Naive GFLOPS   cuBLAS GFLOPS\n";
            std::cout << "-------------------------------------\n";
        }
        
        for (int j = 0; j < numBatches; j++) {
            
            int batch = batchSizes[j];
            
            Result r = runTest(N, batch);

            std::cout << std::fixed << std::setprecision(2);

            if (N <= 32) {
                std::cout << std::setw(6) << batch << std::setw(16) << r.naiveGFLOPS
                << std::setw(16) << r.sharedGFLOPS
                << std::setw(14) << r.sharedSpeedup << "x";
                
                if (N == 32) {
                    std::cout << std::setw(14) << r.coarsenedGFLOPS << std::setw(14) << r.speedup << "x";

                }
                else {
                    std::cout << std::setw(14) << "N/A" <<
                    std::setw(14) << "N/A";
                }
                
                std::cout << std::setw(14) << r.cuBLASGFLOPS << "\n";
            }
            
            else {
                std::cout << std::setw(6) << batch
                << std::setw(16) << r.naiveGFLOPS
                << std::setw(18) << r.cuBLASGFLOPS
                << "\n";
            }
    
        }
    }

    return 0;
}