#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <assert.h>

#define size 256

// parallel sum reduction
template<typename Scalar_type>
__global__ void sum_reduction(Scalar_type* A, Scalar_type* A_reduced){
  // shared memory allocation
  __shared__ Scalar_type A_temporary_sum[size];

  // thread identification
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  // moving values into shared memory
  A_temporary_sum[threadIdx.x] = A[thread_id];

  // syncing threads
  __syncthreads();

  // iterating through the block
  for(int i = 1; i < blockDim.x; i *= 2){
    // using sequential threads
    auto sequential_thread_idx = 2 * i * threadIdx.x;

    // reducing working threads
    if(sequential_thread_idx < blockDim.x){
      A_temporary_sum[sequential_thread_idx] += A_temporary_sum[sequential_thread_idx + i];
    }

    // syncing threads
    __syncthreads();
  }

  // first thread in block writes result
  if(threadIdx.x == 0){
    A_reduced[blockIdx.x] = A_temporary_sum[0];
  }
}

// setting up a vector of ones
template<typename Scalar_type, typename Int_type>
void vector_of_ones(Scalar_type* A, Int_type N){
  for(int i = 0; i < N; ++i){
    A[i] = 1.0;
  }
}

int main(){
  // vector parameters
  size_t N = 1 << 16;
  size_t bytes = N*sizeof(double);

  // host vectors
  double *A_host;
  double *A_reduced_host;

  // device vectors
  double *A_device;
  double *A_reduced_device;

  // allocating memory on host
  A_host = (double*)malloc(bytes);
  A_reduced_host = (double*)malloc(bytes);

  // allocating memory on device
  cudaMalloc(&A_device, bytes);
  cudaMalloc(&A_reduced_device, bytes);

  // setting the vector
  vector_of_ones(A_host, N);

  // copying vector from host to device
  cudaMemcpy(A_device, A_host, bytes, cudaMemcpyHostToDevice);

  // number of threads per block
  const auto number_of_threads = size;

  // number of blocks per grid
  const auto number_of_blocks = (int)ceil(N/number_of_threads);

  // first reduction
  sum_reduction <<<number_of_blocks, number_of_threads>>>(A_device, A_reduced_device);

  // second reduction
  sum_reduction <<<1, number_of_threads>>>(A_reduced_device, A_reduced_device);

  // copying vector from device to host
  cudaMemcpy(A_reduced_host, A_reduced_device, bytes, cudaMemcpyDeviceToHost);

  // output
  std::cout << "The calculated reduced sum is " << A_reduced_host[0] << std::endl;
}



