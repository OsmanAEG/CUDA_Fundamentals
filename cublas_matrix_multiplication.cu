#include <time.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdlib.h>
#include <assert.h>

// checking accuracy of matrix multiplication
template<typename Scalar_type, typename Int_type>
void check_matrix_multiplication(Scalar_type* A, Scalar_type* B, Scalar_type* C, Int_type N){
  Scalar_type c_ij;
  float tolerance = 1.0E-2;

  for(int i = 0; i < N; ++i){
    for(int j = 0; j < N; ++j){
      c_ij = 0.0;
      for(int k = 0; k < N; ++k){
        c_ij += A[k*N + i] * B[j*N + k];
      }
      assert(fabs(C[j*N + i] - c_ij) < tolerance);
    }
  }
}

int main(){
  // matrix dimensional parameters
  int N = 1 << 10;
  size_t bytes = N*N*sizeof(float);

  // input matrices on host memory
  float *A_host;
  float *B_host;

  // output matrices on device memory
  float *C_host;

  // input matrices on device memory
  float *A_device;
  float *B_device;

  // output matrices on device memory
  float *C_device;

  // allocating host memory
  A_host = (float*)malloc(bytes);
  B_host = (float*)malloc(bytes);
  C_host = (float*)malloc(bytes);

  // allocating device memory
  cudaMalloc(&A_device, bytes);
  cudaMalloc(&B_device, bytes);
  cudaMalloc(&C_device, bytes);

  // random number generator
  curandGenerator_t cuda_random_num;
  curandCreateGenerator(&cuda_random_num, CURAND_RNG_PSEUDO_DEFAULT);

  // random number seed
  curandSetPseudoRandomGeneratorSeed(cuda_random_num, (unsigned long long)clock());

  // filling matrices on device with random numbers
  curandGenerateUniform(cuda_random_num, A_device, N*N);
  curandGenerateUniform(cuda_random_num, B_device, N*N);

  // setting the cublas handler
  cublasHandle_t h;
  cublasCreate(&h);

  // scales
  float a = 1.0;
  float b = 0.0;

  // cublas matrix multiplication
  cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &a, A_device, N, B_device, N, &b, C_device, N);

  // copying device memory to host
  cudaMemcpy(A_host, A_device, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(B_host, B_device, bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(C_host, C_device, bytes, cudaMemcpyDeviceToHost);

  // check the answer
  check_matrix_multiplication(A_host, B_host, C_host, N);

  std::cout << "The cublas matrix multiplication worked!" << std::endl;
}