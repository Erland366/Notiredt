#include <stdio.h>

__global__ void testKernel(float* A, unsigned int* B){
    *B = blockIdx.x;
}

void test(float* A_h){
    float* A_d;
    unsigned int* B_h = (unsigned int*)malloc(sizeof(unsigned int));
    unsigned int* B_d;

    cudaMalloc((void**) &A_d, sizeof(unsigned int));
    cudaMalloc((void**) &B_d, sizeof(unsigned int));
    
    cudaMemcpy(A_d, A_h, sizeof(unsigned int), cudaMemcpyHostToDevice);

    testKernel<<<3, sizeof(unsigned int)>>>(A_d, B_d);

    cudaMemcpy(B_h, B_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("Hello : %u", *B_h);

    cudaFree(A_d);
    cudaFree(B_d);
    free(B_h);
}

int main(int argc, char** argv){
    float* X = (float*) malloc(sizeof(unsigned int));

    test(X);

    free(X);
    return 0;
}
