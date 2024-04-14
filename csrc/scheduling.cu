#include <stdio.h>

void incorrect_barrier_example(int n)
{
    char *a = "c";
    printf("%c", a);
    if (a == "c")
    {
        __syncthreads();
    }
    else
    {
        __syncthreads();
    }
}