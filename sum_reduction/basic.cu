#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

#define size 256

// parallel sum reduction
template<typename Scalar_type>
__global__ void sum_reduction(Scalar_type* A, Scalar_type* A_reduced){
  // shared memory
  __shared__ Scalar_type temporary_sum[size];

  // thread ID
  const auto thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  // copying values to shared memory
  temporary_sum[threadIdx.x] = A[thread_id];

  // syncing threads
  __syncthreads();

  // sum every two elements
  for(int i = 1; i < blockDim.x; i *= 2){
    // only half the threads need to do work with each iteration
    if(threadIdx.x % (2*i) == 0){
      temporary_sum[threadIdx.x] += temporary_sum[threadIdx.x + i];
    }
    __syncthreads();
  }

  // thread 0 in this block writes result to main memory
  if(threadIdx.x == 0){
    A_reduced[blockIdx.x] = temporary_sum[0];
  }
}

// setting a vector
template<typename Scalar_type, typename Int_type>
void set_vector(Scalar_type* A, Int_type N){
  for(int i = 0; i < N; ++i){
    A[i] = 1.0;
  }
}

int main(){
  // vector parameters
  int N = 1 << 16;
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
  set_vector(A_host, N);

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
