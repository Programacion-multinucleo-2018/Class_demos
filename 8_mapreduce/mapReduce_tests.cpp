#include <iostream>
#include <vector>
#include <string>

#include "functions.h"
#include "map.h"
#include "reduce.h"
#include "mapreduce.h"

template<class T>
void print_vector(const std::vector<T> &values)
{
    std::cout << "[";

    for (const T &value : values)
    {
        std::cout << " " << value;
    }

    std::cout << " ]" << std::endl;
}

int main(int argc, char **argv)
{
    auto values_a = std::vector<int>( { 1, 2, 3, 4, 5 } );
    auto values_b = std::vector<int>( { 6, 7, 8, 9, 10 } );

    auto animals = std::vector<std::string>({"cat", "dog", "mouse", "bird"});

    // auto result = simpleMap(a, b);
    auto result = map(diff<int>, values_a, values_b);

    print_vector(values_a);
    print_vector(values_b);
    print_vector(result); 
    
    auto total = reduce(diff<int>, result);
    auto strings = reduce(join_strings, animals);

    auto dot = mapReduce(mult<int>, sum<int>, values_b, values_a);

    std::cout << "Total reduction: " << total << std::endl;
    std::cout << "Total dot: " << dot << std::endl;
    std::cout << "Strings: " << strings << std::endl;

    return 0;
}