#include <stdio.h>
#include <cuda_runtime.h>

__global__ void square_array(int* data, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n){
        data[index] += 5;
    }
}

int main(int argc, char** argv){
    int *data;
    int n = 1024;

    // This is where UVM happening, by using cudaMallocManaged
    cudaMallocManaged(&data, n * sizeof(int));

    for (int i = 0; i < n; i++){
        data[i] = i;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    square_array<<<blocksPerGrid, threadsPerBlock>>>(data, n);

    cudaDeviceSynchronize();

    for (int i = 0; i < n; i++){
        data[i] *= 2;
    }

    for (int i = 0; i < 10; i++){
        printf("%d ", data[i]);
        data[i] *= 2;
    }

    printf("\n");

    // So only free on the CUDA?
    cudaFree(data);

    return 0;
};

