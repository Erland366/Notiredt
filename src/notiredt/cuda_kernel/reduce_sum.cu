#include "timer.h"

__global__ void parallelSumReductionKernel(float* A, float* B){
    unsigned int i = 2 * threadIdx.x;
    
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2){
        if (threadIdx.x % stride == 0){
            A[i] += A[i + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
         *B = A[0];
    }
}

void parallelSumReduction(float* A, float* B, unsigned int N){

    float *A_d, *B_d;
    CUDA_SAFE_CALL(cudaMalloc((void**)&A_d, N * sizeof(float)));
    CUDA_SAFE_CALL(cudaMalloc((void**)&B_d, sizeof(float)));
    cudaDeviceSynchronize();

    CUDA_SAFE_CALL(cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(B_d, B, 1 * sizeof(float), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();

    parallelSumReductionKernel<<<1, N>>>(A_d, B_d);
    cudaErr_t err = cudaGetLastError()
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch parallelSumReductionKernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
    CUDA_SAFE_CALL(cudaMemcpy(B, B_d, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(A_d);
    cudaFree(B_d);
}

float sum(float* A, unsigned int N){
    float sum = 0.0f;
    for (unsigned int i = 0; i < N; ++i){
        sum += A[i];
    }
    return sum;
}



int main(int argc, char**argv) {
    unsigned int N = (argc > 1)?(atoi(argv[1])):1024;
    float *A, *B, *sum_cpu;

    A = (float *)malloc(N * sizeof(float));
    B = (float *)malloc(1 * sizeof(float));
    sum_cpu = (float *)malloc(sizeof(float));

    for (unsigned int i = 0; i < N; ++i){
        A[i] = (float)i;
    }

    parallelSumReduction(A, B, N);
    printf("Sum GPU: %f\n", *B);

    *sum_cpu = sum(A, N);
    printf("Sum CPU: %f\n", *sum_cpu);
    
    free(A);
    free(B);
    free(sum_cpu);
}