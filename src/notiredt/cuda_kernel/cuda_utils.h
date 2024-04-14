#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess){ \
            fprintf(stderr, "CUDA ERROR : %s in %s at line %d \n", \
                    cudaGetErrorString(err), __FILE__, __LINE__ \
                    ); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)
