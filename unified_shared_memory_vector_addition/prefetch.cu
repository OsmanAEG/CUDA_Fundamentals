#include <stdio.h>
#include <cassert>
#include <iostream>

// parallel vector addition
template<typename Scalar_type, typename Int_type>
__global__ void vector_addition(const Scalar_type* A, const Scalar_type* B,
                                Scalar_type* C, const Int_type N){
  // global thread id
  auto thread_id = (blockDim.x * blockIdx.x) + threadIdx.x;

  // checking if within boundary
  if(thread_id < N){
    C[thread_id] = A[thread_id] + B[thread_id];
  }
}

// checking addition results on host
template<typename Vector_type, typename Int_type>
void vector_addition_check(const Vector_type& A, const Vector_type& B,
                           const Vector_type& C, const Int_type N){
  for(int i = 0; i < N; ++i){
    assert(C[i] == A[i] + B[i]);
  }
}

// main function
int main(){
  constexpr int N = 1 << 26;
  constexpr size_t bytes = sizeof(int)*N;

  // vector pointers for shared memory
  int *A_shared, *B_shared, *C_shared;

  // vectors on unified shared memory
  cudaMallocManaged(&A_shared, bytes);
  cudaMallocManaged(&B_shared, bytes);
  cudaMallocManaged(&C_shared, bytes);

  // getting device id
  int device_id = cudaGetDevice(&device_id);

  // hints
  cudaMemAdvise(A_shared, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
  cudaMemAdvise(B_shared, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

  // prefetching the output vector to the device
  cudaMemPrefetchAsync(C_shared, bytes, device_id);

  // setting random values for input vectors
  for(int i = 0; i < N; ++i){
    A_shared[i] = rand()%100;
    B_shared[i] = rand()%100;
  }

  // hinting to only read input vectors on the device
  cudaMemAdvise(A_shared, bytes, cudaMemAdviseSetReadMostly, device_id);
  cudaMemAdvise(B_shared, bytes, cudaMemAdviseSetReadMostly, device_id);

  // prefetching output vectors to the device
  cudaMemPrefetchAsync(A_shared, bytes, device_id);
  cudaMemPrefetchAsync(B_shared, bytes, device_id);

  // threads per block
  int block_size = 1 << 10;

  // blocks per grid
  int grid_size = (N + block_size - 1)/block_size;

  vector_addition<<<grid_size, block_size>>>(A_shared, B_shared, C_shared, N);

  // wait to complete kernel execution
  cudaDeviceSynchronize();

  // prefetch back to host
  cudaMemPrefetchAsync(A_shared, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(B_shared, bytes, cudaCpuDeviceId);
  cudaMemPrefetchAsync(C_shared, bytes, cudaCpuDeviceId);

  // checking results
  vector_addition_check(A_shared, B_shared, C_shared, N);

  // free allocated memory
  cudaFree(A_shared);
  cudaFree(B_shared);
  cudaFree(C_shared);

  std::cout << "The vector addition results are correct!" << std::endl;

  return 0;
}