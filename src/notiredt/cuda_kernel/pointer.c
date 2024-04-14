#include <stdio.h>
#include <stdlib.h>

void addone(int* n){
    (*n)++;
}

void addone_pp(int** n){
    (**n)++;
}

void generateFibonacci(int *fibArray, int n){
    fibArray[0] = 0;
    fibArray[1] = 1;
    for (unsigned int i = 2; i < n; i++){
        fibArray[i] = fibArray[i - 1] + fibArray[i - 2];
    }
}

int main() {

    int var1 = 3;

    // Define a pointer by using *, and point it to something using &;
    int *pointer_to_a = &var1;
    int **pointer_to_pointer_to_a = &pointer_to_a;

    printf("Now value of var1 variable: %d\n", var1);
    printf("Value of var1 is : %d\n", *pointer_to_a);

    var1 += 1;
    *pointer_to_a += 1;
    // This should result in 5 I think?
    printf("Now value of var1 variable: %d\n", var1);

    addone(&var1);

    printf("Now now value of var1 variable: %d\n", var1);

    addone_pp(&pointer_to_a);
    
    printf("Now now now value of var1 variable: %d\n", var1);

    int n = 10;
    // printf("Fibonacci you want? : ");
    // scanf("%d", &n);

    int *fibArray = (int *)malloc(n * sizeof(int));
    // Memory allocation can failed?!?!
    if (fibArray == NULL){
        printf("Memory allocation failed.\n");
        return 1;
    }

    generateFibonacci(fibArray, n);


    printf("first %d terms of Fibonacci sequence:\n", n);
    for (unsigned int i = 0; i < n; i++){
        printf("%d ", fibArray[i]);
    }
    
    free(fibArray);

    printf("The actual address of pp is %p\n", pointer_to_a);
    printf("The actual address of ppp is %p\n", pointer_to_pointer_to_a);

    return 0;
}
