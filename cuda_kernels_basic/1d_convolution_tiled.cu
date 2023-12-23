#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void convolution1D(float* input, float* kernel, float* output, int inputSize, int kernelSize){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tile[TILE_SIZE];

    float result = 0.0f;
    int halfKernelSize = kernelSize/2;

    for(int i = -halfKernelSize; i<=halfKernelSize; i++){
        int index = tid + i;

        if(index>=0 && index<inputSize){
            tile[threadIdx.x + i] = input[index];
        }
        else{
            tile[threadIdx.x + i] = 0.0f;
        }

        __syncthreads();
    
        if(index >= 0 && index < inputSize){
            result += tile[threadIdx.x + i + halfKernelSize] * kernel[i + halfKernelSize];
        }

        __syncthreads();
    } 

    if(tid < inputSize){
        output[tid] = result;
    }
}

void initializeArray(float* arr, int size){
    for(int i=0; i<size; i++){
        arr[i] = static_cast<float>(rand())/RAND_MAX;
    }
}

int main(){

    int inputSize = 1024;
    int kernelSize = 5;

    float* input = new float[inputSize];
    initializeArray(input, inputSize);

    float* kernel = new float[kernelSize];
    initializeArray(kernel, kernelSize);
    
    float* output = new float[inputSize];

    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, inputSize*sizeof(float));
    cudaMalloc((void**)&d_kernel, inputSize*sizeof(float));
    cudaMalloc((void**)&d_output, inputSize*sizeof(float));

    cudaMemcpy(d_input, input, inputSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = TILE_SIZE;
    int gridSize = (inputSize + blockSize -1) / blockSize;
    convolution1D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, inputSize, kernelSize);

    cudaMemcpy(output, d_output, inputSize * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    delete[] input;
    delete[] kernel;
    delete[] output;

    std::cout<<"Execution Completed"<<std::endl;


    return 0;
}