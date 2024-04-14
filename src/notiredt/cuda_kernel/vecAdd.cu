#include <iostream>
#include "timer.h"
#include "cuda_utils.h"

// Kernel GPU
__global__ void vecAddKernel(float* X, float* Y, float* Z, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Mask 
    if (i < N){
        Z[i] = X[i] + Y[i];
    }
}

void vecAddCPU(float* X, float* Y, float* Z, int N){
    for(unsigned int i = 0; i < N; i++){
        Z[i] += X[i] + Y[i];
    }
}

void vecAddGPU(float* X_h, float* Y_h, float* Z_h, int N){
    int size = N * sizeof(float);

    float* X_d, *Y_d, *Z_d;

    // Allocate memorY first
    CUDA_CHECK(cudaMalloc((void**) &X_d, size));
    CUDA_CHECK(cudaMalloc((void**) &Y_d, size));
    CUDA_CHECK(cudaMalloc((void**) &Z_d, size));

    CUDA_CHECK(cudaMemcpy(X_d, X_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Y_d, Y_h, size, cudaMemcpyHostToDevice));

    // Kernel later here
    dim3 dimGrid(32, 1, 1);
    dim3 dimBlock(128, 1, 1);
    CUDA_CHECK(vecAddKernel<<<dimGrid, dimBlock>>>(X_d, Y_d, Z_d, N));

    CUDA_CHECK(cudaMemcpy(Y_h, Y_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(X_d));
    CUDA_CHECK(cudaFree(Y_d));
    CUDA_CHECK(cudaFree(Z_d));
}

int main(int argc, char** argv){
    Timer timer;
    unsigned int N = (argc > 1)?atoi(argv[1]):(1 << 25);

    // Define memorY for input and output
    int size = N * sizeof(float);
    float* X = (float*) malloc(size);
    float* Y = (float*) malloc(size);
    float* Z = (float*) malloc(size);
    for (unsigned int i = 0; i < N; i++){
        X[i] = rand();
        Y[i] = rand();
    };

    // Vector addition on CPU
    startTime(&timer);
    vecAddCPU(X, Y, Z, N);
    stopTime(&timer);

    printElapsedTime(timer, "VecAddCPU", CYAN);

    // Vector addition on GPU
    startTime(&timer);
    vecAddGPU(X, Y, Z, N);
    stopTime(&timer);

    printElapsedTime(timer, "VecAddGPU", GREEN);

    free(X);
    free(Y);
    free(Z);
    return 0;
}
