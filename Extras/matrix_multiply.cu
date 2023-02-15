#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

// parallel matrix multiply
template<typename Scalar_type, typename Int_type>
__global__ void matrix_multiply(const Scalar_type* A, const Scalar_type* B, Scalar_type* C,
                                const Int_type M, const Int_type N, const Int_type P){
  // getting columns and rows
  auto column = blockIdx.x * blockDim.x + threadIdx.x;
  auto row    = blockIdx.y * blockDim.y + threadIdx.y;

  // initializing the element value
  C[column + row*P] = 0;

  for(int k = 0; k < M; ++k){
    C[column + row*P] += A[k + row*M]*B[k*N+row];
  }
}

// checking matrix multiply results on host
template<typename Vector_type, typename Int_type>
void matrix_multiply_check(const Vector_type& A, const Vector_type& B, const Vector_type& C,
                           const Int_type M, const Int_type N, const Int_type P){
  // rows
  for(int i = 0; i < P; ++i){
    // columns
    for(int j = 0; j < N; ++j){
      int c_ij = 0;
      // each elements being multiplied
      for(int k = 0; k < M; ++k){
        c_ij += A[k + j*M]*B[k*N + j];
      }

      // verifying result
      assert(c_ij == C[i+j*P]);
    }
  }
}

int main(){
  // matrix dimensional values
  int M = 1 << 10;
  int N = 1 << 11;
  int P = 1 << 12;

  // memory on host
  std::vector<int> A_host(M*N);
  std::vector<int> B_host(P*M);
  std::vector<int> C_host(P*N);

  // size of matrices
  size_t A_bytes = M*N*sizeof(int);
  size_t B_bytes = P*M*sizeof(int);
  size_t C_bytes = P*N*sizeof(int);

  // setting random values for input matrices
  std::generate(A_host.begin(), A_host.end(), [](){
    return rand()%100;
  });

  std::generate(B_host.begin(), B_host.end(), [](){
    return rand()%100;
  });

  // memory on device
  int *A_device, *B_device, *C_device;
  cudaMalloc(&A_device, A_bytes);
  cudaMalloc(&B_device, B_bytes);
  cudaMalloc(&C_device, C_bytes);

  // copying host to device memory
  cudaMemcpy(A_device, A_host.data(), A_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(B_device, B_host.data(), B_bytes, cudaMemcpyHostToDevice);

  // number of threads per block
  int number_of_threads = 32;

  // number of blocks per grid
  int number_of_blocks_x = P/number_of_threads;
  int number_of_blocks_y = N/number_of_threads;

  // three dimensional structs
  dim3 threads_3d(number_of_threads, number_of_threads);
  dim3 blocks_3d(number_of_blocks_x, number_of_blocks_y);

  // kernel execution
  matrix_multiply<<<blocks_3d, threads_3d>>>(A_device, B_device, C_device, M, N, P);

  // copying device to host memory
  cudaMemcpy(C_host.data(), C_device, C_bytes, cudaMemcpyDeviceToHost);

  // checking the answers
  matrix_multiply_check(A_host, B_host, C_host, M, N, P);

  std::cout << "The matrix multiply results are correct!" << std::endl;

  // free device memory
  cudaFree(A_device);
  cudaFree(B_device);
  cudaFree(C_device);

  return 0;
}


