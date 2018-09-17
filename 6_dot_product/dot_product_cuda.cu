#include <stdio.h>
#include <stdlib.h>
#include <math.h>

const int n = 1<<22;
const int blockSize = 1024;
const int gridSize = (int)ceil((float)n/blockSize);

__global__ void dotProduct(float *a, float *b, float *c, int n)
{
    __shared__ int cache[blockSize];

    int tId = blockIdx.x*blockDim.x+threadIdx.x;
    int cacheIndex = threadIdx.x;

    if (tId < n)
        cache[cacheIndex] = a[tId] * b[tId];

    __syncthreads();
    
    // For reductions, threadsPerBlock must be a power of 2
    int i = blockDim.x / 2;
    while(i != 0)
    {
        if(cacheIndex < i)
            cache[cacheIndex] += cache[cacheIndex + i];
        __syncthreads();
        i /= 2;
    }

    if(cacheIndex == 0)
        c[blockIdx.x] = cache[0];
}
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    float n_f = (float)n;
 		
    // Host input vectors
    float *h_a, *h_b, *h_c;
     
    // Device input vectors
    float *d_a, *d_b, *d_c;
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(float);
 
    // Allocate memory for each vector on host
    h_a = new float[bytes](); //(float*)malloc(bytes);
    h_b = new float[bytes](); //(float*)malloc(bytes);
    h_c = new float[blockSize * sizeof(float)];
 
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, gridSize * sizeof(float));
 
    // Initialize vectors on host
    for(int i = 0; i <= n; i++ ) {
        h_a[i] = 1;
        h_b[i] = 1;
    }
 
    // Copy host vectors to device
    cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice);
 
    // Execute the kernel
    dotProduct<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy array back to host
    cudaMemcpy( h_c, d_c, gridSize * sizeof(float), cudaMemcpyDeviceToHost );
 
    float sum;
    for(int i=0; i < gridSize; i++)
        sum += h_c[i];

    printf("final result: %f\n", sum / n_f);
    printf("final result: %f\n", sum);
 
    // Release device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
 
    // Release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
