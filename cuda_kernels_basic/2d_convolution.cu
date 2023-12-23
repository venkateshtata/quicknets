#include <cuda_runtime.h>
#include <iostream>

#define TILE_SIZE 16

__global__ void convolution2D(float* input, float* kernel, float* output, int width, int height, int kernelSize){
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.y;
    int by = blockIdx.y;

    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    float result = 0.0f;
    int halfKernelSize = kernelSize/2;

    for(int i = -halfKernelSize; i <= halfKernelSize; i++){
        for(int j = -halfKernelSize; j <= halfKernelSize; j++){
            int row = by * blockDim.y + ty + i;
            int col = bx * blockDim.x + tx + j;

            if(row>=0 && row<height && col>=0 && col<width){
                tile[ty + i][tx + j] = input[row * width + col];
            }else{
                tile[ty + i][tx + j] = 0.0f;
            }
        }
    }
    
    __syncthreads();

    for(int i=0; i<kernelSize; i++){
        for(int j=0; j<kernelSize; j++){
            result += tile[ty + 1][tx + j] * kernel[i * kernelSize + j];
        }
    }

    int outputIndex = by * blockDim.y + ty;
    outputIndex = outputIndex * width + bx * blockDim.x + tx;

    if(outputIndex<width*height){
        output[outputIndex] = result;
    }
}



void initializeArray(float* arr, int size) {
    for (int i = 0; i < size; ++i) {
        arr[i] = static_cast<float>(rand()) / RAND_MAX;  // Random values between 0 and 1
    }
}


int main() {

    int width = 512; 
    int height = 512;
    int kernelSize = 3; 

    float* input = new float[width * height];
    initializeArray(input, width * height);

    float* kernel = new float[kernelSize * kernelSize];
    initializeArray(kernel, kernelSize * kernelSize);

    float* output = new float[width * height];

    
    float *d_input, *d_kernel, *d_output;
    cudaMalloc((void**)&d_input, width * height * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernelSize * kernelSize * sizeof(float));
    cudaMalloc((void**)&d_output, width * height * sizeof(float));

    cudaMemcpy(d_input, input, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    convolution2D<<<gridSize, blockSize>>>(d_input, d_kernel, d_output, width, height, kernelSize);

    cudaMemcpy(output, d_output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    delete[] input;
    delete[] kernel;
    delete[] output;

    std::cout<<"Execution Completed!"<<std::endl;

    return 0;
}