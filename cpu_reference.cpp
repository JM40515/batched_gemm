//CPU reference for correctness

void CPUBatchedGemm(float* A, float* B, float* C, int batchCount, int N) {

    for (int i = 0; i < batchCount; i++) {
        float* Ai = A + i * N * N;

        float* Bi = B + i * N * N;

        float* Ci = C + i * N * N;

        for (int row = 0; row < N; row++) {
            for (int col = 0; col < N; col++) {
                float sum = 0.0f;

                for (int k = 0; k < N; k++) {
                    sum += Ai[row * N + k] * Bi[k * N + col];
                }

                Ci[row * N + col] = sum;
            }
        }
    }
}