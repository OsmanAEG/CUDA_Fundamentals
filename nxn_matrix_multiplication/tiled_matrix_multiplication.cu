#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

// matrix dimensional values
const int N = 1 << 11;
const int shared_memory_size = 1 << 11;

// parallel matrix multiplication
template<typename Scalar_type>
__global__ void matrix_multiply(const Scalar_type* A, const Scalar_type* B, Scalar_type* C){
  // global index for the output matrix
  auto row = blockIdx.y * blockDim.y + threadIdx.y;
  auto col = blockIdx.x * blockDim.x + threadIdx.x;

  // allocating shared memory
  __shared__ Scalar_type A_shared[shared_memory_size];
  __shared__ Scalar_type B_shared[shared_memory_size];

  Scalar_type c_ij = 0.0;

  // loading in tile values
  for(int i = 0; i < N; i += blockDim.x){
    A_shared[threadIdx.y * blockDim.x + threadIdx.x] = A[row*N + i + threadIdx.x];
    B_shared[threadIdx.y * blockDim.x + threadIdx.x] = B[i*N + col + threadIdx.y*N];

    // wait for all tile memory to load in
    __syncthreads();

    // matrix multiplication for the tiled matrix
    for(int j = 0; j < blockDim.x; ++j){
      c_ij += A_shared[threadIdx.y * blockDim.x + j] * B_shared[j*blockDim.x + threadIdx.x];
    }

    // wait for all threads to finish calculations
    __syncthreads();
  }

  C[row*N + col] = c_ij;
}

// check the matrix multiplication
template<typename Vector_type>
void check_matrix_multiply(const Vector_type A, const Vector_type B, const Vector_type C){
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      auto c_ij = 0.0;
      for(int k = 0; k < N; ++k){
        c_ij += A[i*N + k]*B[k*N + j];
      }
      // verifying answer
      assert(std::abs((C[i*N + j] - c_ij)/c_ij) < 1.0E-6);
    }
  }
}

int main(){
  // matrix dimensional values
  size_t bytes = N*N*sizeof(int);

  // input matrices on host memory
  std::vector<int> A_host(N*N);
  std::vector<int> B_host(N*N);

  // output matrix on host memory
  std::vector<int> C_host(N*N);

  // filling input matrices with random values
  std::generate(A_host.begin(), A_host.end(), [](){
    return rand()%100;
  });

  std::generate(B_host.begin(), B_host.end(), [](){
    return rand()%100;
  });

  // input matrices on device memory
  int *A_device, *B_device, *C_device;
  cudaMalloc(&A_device, bytes);
  cudaMalloc(&B_device, bytes);
  cudaMalloc(&C_device, bytes);

  // copying host memory to device memory
  cudaMemcpy(A_device, A_host.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(B_device, B_host.data(), bytes, cudaMemcpyHostToDevice);

  // number of threads per block
  int number_of_threads = 32;

  // number of blocks per grid
  int number_of_blocks = N/number_of_threads;

  // setting up block and grid values
  dim3 threads(number_of_threads, number_of_threads);
  dim3 blocks(number_of_blocks, number_of_blocks);

  // kernel execution for matrix multiply
  matrix_multiply<<<blocks, threads>>>(A_device, B_device, C_device);

  // copying the output matrix from device to host memory
  cudaMemcpy(C_host.data(), C_device, bytes, cudaMemcpyDeviceToHost);

  // verifying answer
  check_matrix_multiply(A_host, B_host, C_host);

  // output the outcome
  std::cout << "Matrix Multiplication Results are Correct!" << std::endl;

  // freeing device memory
  cudaFree(A_device);
  cudaFree(B_device);
  cudaFree(C_device);

  return 0;
}