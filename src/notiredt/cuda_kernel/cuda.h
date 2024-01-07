#define CUDA_SAFE_CALL(x)                                        \
    do {                                                         \
        CUresult result = x;                                     \
        if (result != CUDA_SUCCESS) {                            \
            const char *msg;                                     \
            cuGetErrorName(result, &msg);                        \
            printf("error: %s failed with error %s\n", #x, msg); \
            exit(1);                                             \
        }                                                        \
    } while (0)

