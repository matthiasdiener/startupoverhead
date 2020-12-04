#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>
#include <iostream>

// Compile with
// nvcc -O2 -std=c++11 cuda.cu

__global__ void
empty()
{

}


int
main(int argc, char **argv)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int threadsPerBlock = 256;
    int blocksPerGrid = 16;

    //Warmup
    empty<<<blocksPerGrid, threadsPerBlock>>>();
    empty<<<blocksPerGrid, threadsPerBlock>>>();
    empty<<<blocksPerGrid, threadsPerBlock>>>();
    
    int nRuns = 1000;

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for(int i=0; i<nRuns; i++) {
        empty<<<blocksPerGrid, threadsPerBlock>>>();
    }

    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        
    std::cout << (float)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/nRuns << " microseconds per call" <<std::endl;

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch empty kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}

