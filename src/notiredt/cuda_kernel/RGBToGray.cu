#include <iostream>
#include "timer.h"
#include "cuda_utils.h"
#define CHANNELS 3

// Pin and Pout is char because char maximum number is 255 which is the same as RGB value
__global__ void colorToGrayscaleConversionKernel(char* Pin, char* Pout, int W, int H){
    // Dimension in CUDA (or actually in IRL idk) it's backward. Hence x is the col and y is the row
    unsigned int col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int row = blockDim.y * blockIdx.y + threadIdx.y;

    bool condition = ((col < W) && (row < H));
    if (condition){
        unsigned int grayOffset = row * W + col;
        unsigned int rgbOffset = grayOffset * CHANNELS;

        unsigned char r = Pin[rgbOffset    ];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

void colorToGrayscaleConversionGPU(char* Pin_h, char* Pout_h, int W, int H){
    int size = W * H * sizeof(char);

    char *Pin_d, *Pout_d;

    // Allocate memory first
    CUDA_CHECK(cudaMalloc((void**) &Pin_d, size));
    CUDA_CHECK(cudaMalloc((void**) &Pout_d, size));

    CUDA_CHECK(cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Pout_d, Pout_h, size, cudaMemcpyHostToDevice));

    // Kernel later here
    unsigned int blockSizeNumber = 32;

    dim3 blockSize(blockSizeNumber, blockSizeNumber, 1);
    dim3 gridSize((W + blockSizeNumber - 1) / blockSizeNumber, (H + blockSizeNumber - 1) / blockSizeNumber, 1);
    colorToGrayscaleConversionKernel<<<gridSize, blockSize>>>(Pin_d, Pout_d, W, H);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(Pin_d));
    CUDA_CHECK(cudaFree(Pout_d));
}

int main(int argc, char** argv){
    Timer timer;

    // Define memorY for input and output
    int W = 256;
    int H = 256;
    int size = W * H * sizeof(char);
    char* Pin = (char*) malloc(size);
    char* Pout = (char*) malloc(size);
    for (unsigned int i = 0; i < W * H; i++){
        Pin[i] = rand();
    };

    // Vector addition on CPU
    //startTime(&timer);
    //vecAddCPU(X, Y, Z, N);
    //stopTime(&timer);

    //printElapsedTime(timer, "VecAddCPU", CYAN);

    // Vector addition on GPU
    startTime(&timer);
    colorToGrayscaleConversionGPU(Pin, Pout, W, H);
    stopTime(&timer);

    cudaDeviceSynchronize();

    printElapsedTime(timer, "colorToGrayscaleConversionGPU", GREEN);

    free(Pin);
    free(Pout);
    return 0;
}
