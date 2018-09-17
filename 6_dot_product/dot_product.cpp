#include <iostream>
#include <cstdio>
#include <omp.h>
#include <chrono>

using namespace std;

int dot_product(int *a, int *b, int elements)
{
    int result = 0;
    for (int index=0; index < elements; index++)
        result += (a[index] * b[index]);
    return result;
}

int dot_product_omp(int *a, int *b, int elements)
{    
    int index = 0, chunk = 1000;
    int result = 0.0f;
    omp_set_num_threads(4);
    #pragma omp parallel for default(shared) private(index) schedule(static,chunk) reduction(+:result)  
    for (index=0; index < elements; index++)
        result += (a[index] * b[index]);
    return result;
}

int main ()
{
    int elements = 80000000;

    int *a = new int[elements], *b = new int[elements], result = 0.0f;

    for (int index=0; index < elements; index++)
    {
        a[index] = 1.0;
        b[index] = 2.0;
    }

    auto start_cpu =  chrono::high_resolution_clock::now();
    // result = dot_product(a, b, elements);
    result = dot_product_omp(a, b, elements);
    auto end_cpu =  chrono::high_resolution_clock::now();
    chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

    printf("Final result for %d elements = %d - elapsed: %f ms\n", elements, result, duration_ms.count());

    delete[] a;
    delete[] b;
}