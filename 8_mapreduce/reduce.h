#include <vector>

// Reduce is an idiom used in the context of a parallel programming model to combine multiple vectors into one, using an associative binary operator.

/**
 * Reduces a vector using a specified funcion.
 * 
 * @param   func    The function used to reduce the vector.
 * @param   a       Vector of type T.
 * @return          A T with the result of the reduction of pairwise elements.
**/
template<class F, class T>
T reduce(F func, const std::vector<T> values)
{
    if(values.empty())
        return T();
    else
    {
        T result = values[0];

        for(size_t idx = 1; idx < values.size(); idx++)
            result = func(result, values[idx]);
        
        return result;
    }
}
