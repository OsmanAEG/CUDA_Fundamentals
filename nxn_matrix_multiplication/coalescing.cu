#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

// select transpose signature
#define transpose_signature 2

// parallel matrix multiplication that takes in a transposed matrix
template<typename Scalar_type, typename Int_type>
__global__ void transpose_matrix_multiply(const Scalar_type* A, const Scalar_type* B,
                                          Scalar_type* C, const Int_type N){
  // thread row and column
  auto row = blockIdx.y * blockDim.y + threadIdx.y;
  auto col = blockIdx.x * blockDim.x + threadIdx.x;

  Scalar_type c_ij = 0.0;

  // enforcing boundary condition
  if((row < N) && (col < N)){
    for(int k = 0; k < N; ++k){
      if constexpr(transpose_signature == 0){
        c_ij += A[row*N + k] * B[k*N + col];
      }
      else if constexpr(transpose_signature == 1){
        c_ij += A[k*N + row] * B[k*N + col];
      }
      else if constexpr(transpose_signature == 2){
        c_ij += A[row*N + k] * B[k + col*N];
      }
      else{
        // throw CUDA runtime error
        assert(0);
      }
    }
    // element answer
    C[row*N + col] = c_ij;
  }
}

// confirm matrix multiply
template<typename Vector_type, typename Int_type>
void confirm_matrix_multiply(const Vector_type A, const Vector_type B,
                             const Vector_type C, const Int_type N){
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      auto c_ij = 0.0;
      for(int k = 0; k < N; ++k){
        c_ij += A[i*N + k]*B[k*N+j];
      }
      assert(std::abs((C[i*N + j] - c_ij)/c_ij) < 1.0E-6);
    }
  }
}

// matrix transpose
template<typename Vector_type, typename Int_type>
void matrix_transpose(Vector_type P, Vector_type& Pt, Int_type N){
  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      Pt[j*N + i] = P[i*N + j];
    }
  }
}

int main(){
  // matrix parameters
  int N = 1 << 10;
  size_t bytes = N*N*sizeof(int);

  // input matrices on host memory
  std::vector<int> A_host(N*N);
  std::vector<int> B_host(N*N);

  // transposed input matrices on host memory
  std::vector<int> At_host(N*N);
  std::vector<int> Bt_host(N*N);

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
  int *A_device, *B_device;
  cudaMalloc(&A_device, bytes);
  cudaMalloc(&B_device, bytes);

  // output matrix on device memory
  int *C_device;
  cudaMalloc(&C_device, bytes);

  // evaluating transposed matrices
  matrix_transpose(A_host, At_host, N);
  matrix_transpose(B_host, Bt_host, N);

  if constexpr(transpose_signature == 0){
    cudaMemcpy(A_device, A_host.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host.data(), bytes, cudaMemcpyHostToDevice);
  }
  else if constexpr(transpose_signature == 1){
    cudaMemcpy(A_device, At_host.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host.data(), bytes, cudaMemcpyHostToDevice);
  }
  else if constexpr(transpose_signature == 2){
    cudaMemcpy(A_device, A_host.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, Bt_host.data(), bytes, cudaMemcpyHostToDevice);
  }
  else{
    // throw CUDA runtime error
    assert(0);
  }

  // number of threads per block
  int number_of_threads = 32;

  // number of blocks per grid
  int number_of_blocks = N/number_of_threads;

  // setting up block and grid values
  dim3 threads(number_of_threads, number_of_threads);
  dim3 blocks(number_of_blocks, number_of_blocks);

  // kernel execution for matrix multiply
  transpose_matrix_multiply<<<blocks, threads>>>(A_device, B_device, C_device, N);

  // copying the output matrix from device to host memory
  cudaMemcpy(C_host.data(), C_device, bytes, cudaMemcpyDeviceToHost);

  // verifying answer
  confirm_matrix_multiply(A_host, B_host, C_host, N);

  // output the outcome
  std::cout << "Matrix Multiplication Results are Correct!" << std::endl;

  // freeing device memory
  cudaFree(A_device);
  cudaFree(B_device);
  cudaFree(C_device);

}