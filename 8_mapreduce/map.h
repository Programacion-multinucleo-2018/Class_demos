#include <vector>

// Map is an idiom in parallel computing where a simple operation is applied to all elements of a sequence, potentially in parallel. It is used to solve embarrassingly parallel problems: those problems that can be decomposed into independent subtasks, requiring no communication/synchronization between the subtasks except a join or barrier at the end.

/**
 * Maps the function sum to two vectors; ie, sums the vectors and returns a vector with the result of each parwise sum.
 * 
 * @param   a   Vector of type T.
 * @param   b   Vector of type T.
 * @return      A vector of type T with the sum of pairwise elements.
**/
template<class T>
std::vector<T> simpleMap(const std::vector<T> a, const std::vector<T> b)
{
    auto result = std::vector<int>( a.size() );

    for (int i=0; i<a.size(); ++i)
    {
        result[i] = sum( a[i], b[i] );
    }

    return result;
}

/**
 * Maps a function to two vectors; ie, aplies the function the vectors and returns a vector with the result of each pairwise function.
 * 
 * @param   func    A function to be applied to the vectors.
 * @param   a       Vector of type T.
 * @param   b       Vector of type T.
 * @return          A vector of type T with the result of the application of the function func to pairwise elements.
**/
template<class F, class T>
std::vector<T> map(F func, const std::vector<T> a, const std::vector<T> b)
{
    int nvalues = std::min( a.size(), b.size() );

    auto result = std::vector<T>(nvalues); 

    for (int i=0; i<nvalues; i++)
    {
        result[i] = func(a[i], b[i]);
    }

    return result;
}