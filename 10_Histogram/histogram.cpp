#include <iostream>
#include <fstream>
#include <omp.h>
#include <chrono>

using namespace std;

void calculateLetterHistogram(char* data, int length, int *histogram)
{
    // #pragma omp parallel for
    for(int i = 0; i<length; i++)
    {
        // #pragma omp atomic
        histogram[(int)data[i]-32] ++;
    }
}

int main()
{
    const char *filename = "../Resources/warandpeace.txt";

    int numBytes = 1048576 * 4;
    char *text = new char[numBytes];
    int *histogram = new int[96]();

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

    inFile.read(text, length);

    auto start_cpu = chrono::high_resolution_clock::now();
    calculateLetterHistogram(text, length, histogram);
    auto end_cpu = chrono::high_resolution_clock::now();

    chrono::duration<float, std::milli> duration_cpu = end_cpu - start_cpu;

    std::cout << "Time spent: " << duration_cpu.count()<<  " ms. Result:" << std::endl;

    for(int i = 0; i< 96; i++)
    {
        cout << histogram[i] << ":" << (char)(i+32) << endl;
    }
}