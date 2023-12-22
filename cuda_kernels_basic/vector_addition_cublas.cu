#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <assert.h>
#include <math.h>
#include<iostream>

void vector_init(float *a, int n){
    for(int i=0; i<n; i++){
        a[i] = (float)(rand()%100);
    }
}

int main(){

    int n=1<<2;
    size_t bytes = n * sizeof(float);

    float *h_a, *h_b, *h_c;
    float *d_a, *d_b;

    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    vector_init(h_a, n);
    vector_init(h_b, n);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
    cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);

    //Laching Saxpy kernel (single precision a * x + y)
    const float scale = 2.0f;
    cublasSaxpy(handle, n, &scale, d_a, 1, d_b, 1);

    cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);

    cublasDestroy(handle);

    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);

    printf("Execution Compelted!\n");

    return 0;




}