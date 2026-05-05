all:
	nvcc main.cu tests.cu kernels.cu cpu_reference.cpp -lcublas -o batched_gemm

clean:
	rm -f batched_gemm