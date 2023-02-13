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

  // setting random values for input vectors
  for(int i = 0; i < N; ++i){
    A_shared[i] = rand()%100;
    B_shared[i] = rand()%100;
  }

  // threads per block
  int number_of_threads = 1 << 10;

  // number of blocks
  int number_of_blocks = (N+ number_of_threads - 1)/number_of_threads;

  vector_addition<<<number_of_blocks, number_of_threads>>>(A_shared, B_shared, C_shared, N);

  // wait to complete kernel execution
  cudaDeviceSynchronize();

  // checking results
  vector_addition_check(A_shared, B_shared, C_shared, N);

  // free allocated memory
  cudaFree(A_shared);
  cudaFree(B_shared);
  cudaFree(C_shared);

  std::cout << "The vector addition results are correct!" << std::endl;

  return 0;
}

