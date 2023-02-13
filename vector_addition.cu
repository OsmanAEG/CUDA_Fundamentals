#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

// vector addition on device
template<typename Scalar_type, typename Int_type>
__global__ void vector_addition(const Scalar_type* A, const Scalar_type* B,
                                Scalar_type* C, const Int_type N){

  int global_thread_id = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(global_thread_id < N){
    C[global_thread_id] = A[global_thread_id] + B[global_thread_id];
  }
}

// check results on host
template<typename Vector_type>
void check(Vector_type& A, Vector_type& B, Vector_type& C){
  for(int i = 0; i < A.size(); ++i){
    assert(C[i] == A[i] + B[i]);
  }
}

int main(){
  constexpr int N = 1 << 16;
  constexpr size_t bytes = sizeof(int)*N;

  // host memory
  std::vector<int> A_host(N);
  std::vector<int> B_host(N);
  std::vector<int> C_host(N);

  // fill the input vectors
  for(int i = 0; i < N; ++i){
    A_host[i] = rand()%100;
    B_host[i] = rand()%100;
  }

  // device memory
  int *A_device, *B_device, *C_device;

  cudaMalloc(&A_device, bytes);
  cudaMalloc(&B_device, bytes);
  cudaMalloc(&C_device, bytes);

  // copying host to device memory
  cudaMemcpy(A_device, A_host.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(B_device, B_host.data(), bytes, cudaMemcpyHostToDevice);

  // device parameters
  int number_of_threads = 1 << 10;
  int number_of_blocks = (N + number_of_threads - 1)/number_of_threads;

  // kernel execution
  vector_addition<<<number_of_blocks, number_of_threads>>>(A_device, B_device, C_device, N);

  // copying device to host memory
  cudaMemcpy(C_host.data(), C_device, bytes, cudaMemcpyDeviceToHost);

  // verification
  check(A_host, B_host, C_host);

  // free device memory
  cudaFree(A_device);
  cudaFree(B_device);
  cudaFree(C_device);

  std::cout << "The Vector Addition Worked!" << std::endl;

  return 0;
}


