#include "../common/common.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <omp.h>

using namespace std;

template <typename T, typename Predicate>
__device__ void count_if(int *count, T *data, int n, Predicate p)
{ 
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; 
         i < n; 
         i += gridDim.x * blockDim.x) 
    {
        if (p(data[i])) 
            atomicAdd(count, 1);
    }
}

__global__
void letter_frequency(int *count, char *text, int n)
{
    const char letters[] { 'x','y','z','w' };

    count_if(count, text, n, [&](char c) {
        for (const auto x : letters) 
            if (c == x) return true;
        return false;
    });
}

void letter_frequency_cpu(int &count, char* text, int n)
{
    const char letters[] { 'x','y','z','w' };

    #pragma omp parallel for schedule(static)
    for(int i = 0; i<n; i++)
    {
        char c = text[i];
        for(const auto x : letters)
        {
            if(c == x)
            // #pragma omp critical
            // This restricts the code so that only one thread can do something at a time
            #pragma omp atomic
            // The atomic directive only applies to memory read/write operations
                count++;
        }
    }

}

int main(int argc, char** argv)
{ 
    const char *filename = "../Resources/warandpeace.txt";

    int numBytes = 1048576 * 4;
    char *h_text = new char[numBytes];

    char *d_text;
    SAFE_CALL(cudaMalloc(&d_text, numBytes), "Error allocating");

    ifstream inFile (filename, ios::binary);

    if (!inFile)
    {
        cout << "Cannot find the input text file\n. Exiting..\n";
        return 0;
    }

    inFile.seekg (0, inFile.end);
    int length = inFile.tellg();

    inFile.seekg (0, inFile.beg);

    cout << "File length " << length << endl;

    inFile.read(h_text, length);
    
    SAFE_CALL(cudaMemcpy(d_text, h_text, length, cudaMemcpyHostToDevice), "Error copying file");
  
    int count = 0;
    int *d_count;
    SAFE_CALL(cudaMalloc(&d_count, sizeof(int)), "Error allocating device count");
    SAFE_CALL(cudaMemset(d_count, 0, sizeof(int)), "Error setting device count");

    const int blockSize = 256;
    const int gridSize = 8;
    // const int gridSize = (int)ceil((float)length/blockSize);
    
    auto start_cpu = chrono::high_resolution_clock::now();
    letter_frequency_cpu(count, h_text, length);
    auto end_cpu = chrono::high_resolution_clock::now();

    chrono::duration<float, std::milli> duration_cpu = end_cpu - start_cpu;

    std::cout << "Counted " << count << " instances of 'x', 'y', 'z', or 'w' in \"" << filename << "\" in " << duration_cpu.count()<<  " ms." << std::endl;

    auto start_gpu =  chrono::high_resolution_clock::now();
    letter_frequency<<<gridSize, blockSize>>>(d_count, d_text, length);
    SAFE_CALL(cudaDeviceSynchronize(), "Error with kernel call");
    auto end_gpu =  chrono::high_resolution_clock::now();

    SAFE_CALL(cudaMemcpy(&count, d_count, sizeof(int), cudaMemcpyDeviceToHost), "Error copying count");
    
    chrono::duration<float, std::milli> duration_gpu = end_gpu - start_gpu;

    std::cout << "Counted " << count << " instances of 'x', 'y', 'z', or 'w' in \"" << filename << "\" in " << duration_gpu.count()<<  " ms." << std::endl;

    SAFE_CALL(cudaFree(d_count),"");
    SAFE_CALL(cudaFree(d_text),"");
}
