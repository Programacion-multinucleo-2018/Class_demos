#include <string>

template<class T>
T sum(T x, T y)
{
    return x + y;
}

template<class T>
T diff(T x, T y)
{
    return x - y;
}

template<class T>
T mult(T x, T y)
{
    return x * y;
}

std::string join_strings(std::string x, std::string y)
{
    return x + " " + y;
}