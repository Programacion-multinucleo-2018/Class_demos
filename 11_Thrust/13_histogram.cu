#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>
#include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include <iostream>
#include <iomanip>
#include <iterator>

// simple routine to print contents of a vector
template <typename Vector>
void print_vector(const std::string& name, const Vector& v)
{
  typedef typename Vector::value_type T;
  std::cout << "  " << std::setw(20) << name << "  ";
  thrust::copy(v.begin(), v.end(), std::ostream_iterator<T>(std::cout, " "));
  std::cout << std::endl;
}

// dense histogram using binary search
template <typename Vector1, 
          typename Vector2>
void dense_histogram(const Vector1& input,
                           Vector2& histogram)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector2::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);
    
  // print the initial data
  print_vector("initial data", data);

  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());
  
  // print the sorted data
  print_vector("sorted data", data);

  // number of histogram bins is equal to the maximum value plus one
  IndexType num_bins = data.back() + 1;

  // resize histogram storage
  histogram.resize(num_bins);
  
  // find the end of each bin of values
  thrust::counting_iterator<IndexType> search_begin(0);
  thrust::upper_bound(data.begin(), data.end(),
                      search_begin, search_begin + num_bins,
                      histogram.begin());
  
  // print the cumulative histogram
  print_vector("cumulative histogram", histogram);

  // compute the histogram by taking differences of the cumulative histogram
  thrust::adjacent_difference(histogram.begin(), histogram.end(),
                              histogram.begin());

  // print the histogram
  print_vector("histogram", histogram);
}

// sparse histogram using reduce_by_key
template <typename Vector1,
          typename Vector2,
          typename Vector3>
void sparse_histogram(const Vector1& input,
                            Vector2& histogram_values,
                            Vector3& histogram_counts)
{
  typedef typename Vector1::value_type ValueType; // input value type
  typedef typename Vector3::value_type IndexType; // histogram index type

  // copy input data (could be skipped if input is allowed to be modified)
  thrust::device_vector<ValueType> data(input);
    
  // print the initial data
  print_vector("initial data", data);

  // sort data to bring equal elements together
  thrust::sort(data.begin(), data.end());
  
  // print the sorted data
  print_vector("sorted data", data);

  // number of histogram bins is equal to number of unique values (assumes data.size() > 0)
  IndexType num_bins = thrust::inner_product(data.begin(), data.end() - 1,
                                             data.begin() + 1,
                                             IndexType(1),
                                             thrust::plus<IndexType>(),
                                             thrust::not_equal_to<ValueType>());

  // resize histogram storage
  histogram_values.resize(num_bins);
  histogram_counts.resize(num_bins);
  
  // compact find the end of each bin of values
  thrust::reduce_by_key(data.begin(), data.end(),
                        thrust::constant_iterator<IndexType>(1),
                        histogram_values.begin(),
                        histogram_counts.begin());
  
  // print the sparse histogram
  print_vector("histogram values", histogram_values);
  print_vector("histogram counts", histogram_counts);
}

int main(void)
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist(0, 9);

  const int N = 40;
  const int S = 4;

  // generate random data on the host
  thrust::host_vector<int> input(N);
  for(int i = 0; i < N; i++)
  {
    int sum = 0;
    for (int j = 0; j < S; j++)
      sum += dist(rng);
    input[i] = sum / S;
  }

  // demonstrate dense histogram method
  {
    std::cout << "Dense Histogram" << std::endl;
    thrust::device_vector<int> histogram;
    dense_histogram(input, histogram);
  }
  
  // demonstrate sparse histogram method
  {
    std::cout << "Sparse Histogram" << std::endl;
    thrust::device_vector<int> histogram_values;
    thrust::device_vector<int> histogram_counts;
    sparse_histogram(input, histogram_values, histogram_counts);
  }

  // Note: 
  // A dense histogram can be converted to a sparse histogram
  // using stream compaction (i.e. thrust::copy_if).
  // A sparse histogram can be expanded into a dense histogram
  // by initializing the dense histogram to zero (with thrust::fill)
  // and then scattering the histogram counts (with thrust::scatter).

  return 0;
}
