#include <cstdio>
#include <chrono>
#include "../common/common.h"

__global__ void reverseNotShared(int *values, int *results, int num_elements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < num_elements)
    {
        int new_id = num_elements - tid -1;

        results[new_id] = values[tid];
    }
}


__global__ void dynamicReverse(int *values, int *results, int num_elements)
{
    extern __shared__ int s[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < num_elements)
    {
        int new_id = num_elements - tid - 1;
        s[threadIdx.x] = values[tid];
        __syncthreads();

        values[new_id] = s[threadIdx.x];
    }
}

__global__ void staticReverse(int *values, int *results, int num_elements)
{
    __shared__ int s[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < num_elements)
    {
        int new_id = num_elements - tid - 1;
        s[threadIdx.x] = values[tid];
        __syncthreads();

        values[new_id] = s[threadIdx.x];
    }
}

void checkResults(const int* result, const int* reverse, int num_elements)
{
    for (int i = 0; i < num_elements; i++) 
        if (result[i] != reverse[i]) 
        {
            printf("Error: v[%d]!=r[%d] (%d, %d)\n", i, i, result[i], reverse[i]);
            return;
        }
}

int main(void)
{
    const int num_elements = 1<<20;
    const int blockSize = 256;
    const int gridSize = (int)ceil((float)num_elements/blockSize);

    printf("Num elements: %d Gridsize: %d Blocksize: %d\n", num_elements, gridSize, blockSize);

    int *values = new int[num_elements], *reverse = new int[num_elements], *result = new int[num_elements]();

    for (int i = 0; i < num_elements; i++) 
    {
        values[i] = i;
        reverse[i] = num_elements-i-1;
    }

    int *d_values, *d_results;
    SAFE_CALL(cudaMalloc(&d_values, num_elements * sizeof(int)), "Error assigning memory"); 
    SAFE_CALL(cudaMalloc(&d_results, num_elements * sizeof(int)), "Error assigning memory"); 

    SAFE_CALL(cudaMemcpy(d_values, values, num_elements*sizeof(int), cudaMemcpyHostToDevice), "Error copying memory to device");

    auto start_cpu =  std::chrono::high_resolution_clock::now();
    reverseNotShared<<<gridSize, blockSize>>>(d_values, d_results, num_elements);
    SAFE_CALL(cudaDeviceSynchronize(), "Error with kernel call");
    auto end_cpu =  std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("Reverse finished. Time spent for version with no shared memory: %.5f ms\n", duration_ms.count());

    SAFE_CALL(cudaMemcpy(result, d_results, num_elements*sizeof(int), cudaMemcpyDeviceToHost), "Error copying memory from device");

    checkResults(result, reverse, num_elements);

    // run version with static shared memory
    start_cpu =  std::chrono::high_resolution_clock::now();
    staticReverse<<<gridSize,blockSize>>>(d_values, d_results, num_elements);
    SAFE_CALL(cudaDeviceSynchronize(), "Error with kernel call");
    end_cpu =  std::chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;

    SAFE_CALL(cudaMemcpy(result, d_results, num_elements*sizeof(int), cudaMemcpyDeviceToHost), "Error copying memory from device");

    printf("Static reverse. Time spent for version with static shared memory: %.5f ms\n", duration_ms.count());
  
    checkResults(result, reverse, num_elements);

    // run dynamic shared memory version

    start_cpu =  std::chrono::high_resolution_clock::now();
    dynamicReverse<<<gridSize,blockSize,blockSize*sizeof(int)>>>(d_values, d_results, num_elements);
    SAFE_CALL(cudaDeviceSynchronize(), "Error with kernel call");
    end_cpu =  std::chrono::high_resolution_clock::now();
    duration_ms = end_cpu - start_cpu;

    SAFE_CALL(cudaMemcpy(result, d_results, num_elements*sizeof(int), cudaMemcpyDeviceToHost), "Error copying memory from device");

    printf("Dynamic reverse correct. Time spent for version with no shared memory: %.5f ms\n", duration_ms.count());

    checkResults(result, reverse, num_elements);

    cudaFree(d_values);
    free(values);
    free(reverse);
    free(result);
}

// includes, system
// #include <stdio.h>
// #include <assert.h>
 
// // Simple utility function to check for CUDA runtime errors
// void checkCUDAError(const char* msg);
 
// // Part 2 of 2: implement the fast kernel using shared memory
// __global__ void reverseArrayBlock(int *d_out, int *d_in)
// {
//     extern __shared__ int s_data[];
 
//     int inOffset  = blockDim.x * blockIdx.x;
//     int in  = inOffset + threadIdx.x;
 
//     // Load one element per thread from device memory and store it 
//     // *in reversed order* into temporary shared memory
//     s_data[blockDim.x - 1 - threadIdx.x] = d_in[in];
 
//     // Block until all threads in the block have written their data to shared mem
//     __syncthreads();
 
//     // write the data from shared memory in forward order, 
//     // but to the reversed block offset as before
 
//     int outOffset = blockDim.x * (gridDim.x - 1 - blockIdx.x);
 
//     int out = outOffset + threadIdx.x;
//     d_out[out] = s_data[threadIdx.x];
// }
 
// ////////////////////////////////////////////////////////////////////////////////
// // Program main
// ////////////////////////////////////////////////////////////////////////////////
// int main( int argc, char** argv) 
// {
//     // pointer for host memory and size
//     int *h_a;
//     int dimA = 256 * 1024; // 256K elements (1MB total)
 
//     // pointer for device memory
//     int *d_b, *d_a;
 
//     // define grid and block size
//     int numThreadsPerBlock = 256;
 
//     // Compute number of blocks needed based on array size and desired block size
//     int numBlocks = dimA / numThreadsPerBlock;  
 
//     // Part 1 of 2: Compute the number of bytes of shared memory needed
//     // This is used in the kernel invocation below
//     int sharedMemSize = numThreadsPerBlock * sizeof(int);
 
//     // allocate host and device memory
//     size_t memSize = numBlocks * numThreadsPerBlock * sizeof(int);
//     h_a = (int *) malloc(memSize);
//     cudaMalloc( (void **) &d_a, memSize );
//     cudaMalloc( (void **) &d_b, memSize );
 
//     // Initialize input array on host
//     for (int i = 0; i < dimA; ++i)
//     {
//         h_a[i] = i;
//     }
 
//     // Copy host array to device array
//     cudaMemcpy( d_a, h_a, memSize, cudaMemcpyHostToDevice );
 
//     // launch kernel
//     dim3 dimGrid(numBlocks);
//     dim3 dimBlock(numThreadsPerBlock);
//     reverseArrayBlock<<< dimGrid, dimBlock, sharedMemSize >>>( d_b, d_a );
 
//     // block until the device has completed
//     cudaThreadSynchronize();
 
//     // check if kernel execution generated an error
//     // Check for any CUDA errors
//     checkCUDAError("kernel invocation");
 
//     // device to host copy
//     cudaMemcpy( h_a, d_b, memSize, cudaMemcpyDeviceToHost );
 
//     // Check for any CUDA errors
//     checkCUDAError("memcpy");
 
//     // verify the data returned to the host is correct
//     for (int i = 0; i < dimA; i++)
//     {
//         assert(h_a[i] == dimA - 1 - i );
//     }
 
//     // free device memory
//     cudaFree(d_a);
//     cudaFree(d_b);
 
//     // free host memory
//     free(h_a);
 
//     // If the program makes it this far, then the results are correct and
//     // there are no run-time errors.  Good work!
//     printf("Correct!\n");
 
//     return 0;
// }
 
// void checkCUDAError(const char *msg)
// {
//     cudaError_t err = cudaGetLastError();
//     if( cudaSuccess != err) 
//     {
//         fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
//         exit(EXIT_FAILURE);
//     }                         
// }