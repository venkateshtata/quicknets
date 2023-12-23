#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

using std::cout;
using std:: generate;
using std::vector;

__global__ void reduce0(int *g_idata, int *g_odata){

    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = g_idata[i];

    __syncthreads();

    for(unsigned int s=1; s<blockDim.x; s*=2){
        if(tid%(2*s)==0){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if(tid==0){
        g_odata[blockIdx.x] = sdata[0];
    }

}

int main(){

    constexpr int N = 1<<16;

    size_t bytes = N * sizeof(int);

    vector<int> h_a(N*N);

    vector<int> h_idata;
    h_idata.reserve(N);

    vector<int> h_odata;
    h_odata.reserve(N);

    for(int i=0; i<N; i++){
        h_idata.push_back(rand()%100);
    }

    int *d_idata, *d_odata;

    cudaMalloc(&d_idata, bytes);
    cudaMalloc(&d_odata, bytes);

    cudaMemcpy(d_idata, h_idata.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_odata, h_odata.data(), bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 1 << 10;
    int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

    reduce0<<<NUM_BLOCKS, NUM_THREADS>>>(d_idata, d_odata);

    cudaFree(d_idata);
    cudaFree(d_odata);


    std::cout << "COMPLETED SUCCESSFULLY\n";

    return 0;







}