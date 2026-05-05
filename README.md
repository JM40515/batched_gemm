# Batched GEMM in Cuda

## Project
This project was intended to implement batched GEMM via Cuda, and more specific to compare
against the strided batched GEMM in the cuBLAS library. I used the RTX3080 GPU in the class server.
What my project has are:

1. A CPU reference 
2. A Naive CUDA Kernel
3. A CUDA kernel that utilizes Shared Memory
4. A CUDA kernel that follows up on the previous by adding thread coarsening (This is the optimal)
5. cuBLAS StridedBatchedGEMM kernel to compare against

## How to Build
This was made in the class server, so connect to UH VPN as we would during the HW assignments.

In order to run the project, type and enter in the terminal:

make

This should load up batched_gemm which is the name of the code.

To test, type and enter in the terminal:

./batched_gemm

and that should do it!

I will also be zipping this up and sending it over on Canvas, and so:
If not working, please feel free to send me a message on Canvas or my Outlook
and I will respond pretty fast.

