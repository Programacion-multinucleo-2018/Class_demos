#include <vector>

// A MapReduce program is composed of a map procedure, and a reduce method, which performs a summary operation (such as counting the number of students in each queue, yielding name frequencies).

/**
 * Applies a map reduce to two vectors.
 * 
 * @param mFunc The function used to map.
 * @param rFunc The function used to reduce.
 * @param a     Vector of type T.
 * @param b     Vector of type T.
 * @return      A value of type T with the result of the map and reduce.
 **/
template<class MAPF, class REDF, class T>
T mapReduce(MAPF mfunc, REDF rfunc, const std::vector<T> a, const std::vector<T> b)
{
    int nvalues = std::min( a.size(), b.size() );

    auto values = std::vector<T>(nvalues); 

    for (int i=0; i<nvalues; i++)
    {
        values[i] = mfunc(a[i], b[i]);
    }

    if(values.empty())
        return T();
    else
    {
        T result = values[0];

        for(size_t idx = 1; idx < values.size(); idx++)
            result = rfunc(result, values[idx]);
        
        return result;
    }
}