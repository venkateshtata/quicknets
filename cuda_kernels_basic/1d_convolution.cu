#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void convolution_1d(int *array, int *mask, int *result, int n, int m){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int r = m/2;

    int start = tid - r;

    int temp = 0;
    
    for(int j=0; j<m; j++){
        if(((start+j)>=0) && (start + j < n)){
            temp += array[start + j] * mask[j];
        }
    }

    result[tid] = temp;
}

int main(){

    int n = 1 << 20;

    int bytes_n = n * sizeof(int);

    int m = 7;

    int bytes_m = m * sizeof(int);

    std::vector<int> h_array(n);
    std::generate(begin(h_array), end(h_array), [](){ return rand()%100;});

    std::vector<int> h_mask(n);
    std::generate(begin(h_mask), end(h_mask), [](){ return rand()%100;});

    std::vector<int> h_result(n);

    int *d_array, *d_mask, *d_result;
    cudaMalloc(&d_array, bytes_n);
    cudaMalloc(&d_mask, bytes_m);
    cudaMalloc(&d_result, bytes_n);

    cudaMemcpy(d_array, h_array.data(), bytes_n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask.data(), bytes_m, cudaMemcpyHostToDevice);

    int THREADS = 256;

    int GRID = (n + THREADS - 1) / THREADS;

    convolution_1d<<<GRID, THREADS>>>(d_array, d_mask, d_result, n, m);

    cudaMemcpy(h_result.data(), d_result, bytes_n, cudaMemcpyDeviceToHost);

    std::cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_result);
    cudaFree(d_mask);
    cudaFree(d_array);

    return 0;

}
