#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <iostream>

// initialize a vector
template<typename Scalar_type, typename Int_type>
void initialize_vector(Scalar_type *P, Int_type N){
  for(int i = 0; i < N; ++i){
    P[i] = (Scalar_type)(rand()%100);
  }
}

// check the vector addition for accuracy
template<typename Scalar_type, typename Int_type>
void check_vector_addition(Scalar_type* A, Scalar_type* B, Scalar_type* C,
                           Scalar_type alpha, Int_type N){
  for(int i = 0; i < N; ++i){
    auto c_i = alpha*A[i] + B[i];
    assert(std::abs((C[i] - c_i)/c_i) < 1.0E-6);
  }
}

int main(){
  // vector parameters
  int N = 1 << 10;
  size_t bytes = N*sizeof(float);

  // input vectors on host memory
  float *A_host = (float*)malloc(bytes);
  float *B_host = (float*)malloc(bytes);

  // output vector on host memory
  float *C_host = (float*)malloc(bytes);

  // input vectors on device memory
  float *A_device, *B_device;
  cudaMalloc(&A_device, bytes);
  cudaMalloc(&B_device, bytes);

  // output vector on device memory
  float *C_device;
  cudaMalloc(&C_device, bytes);

  // filling input vectors with random values
  initialize_vector(A_host, N);
  initialize_vector(B_host, N);

  // cublas handler
  cublasHandle_t h;
  cublasCreate_v2(&h);

  // transfer host memory to device memory
  cublasSetVector(N, sizeof(float), A_host, 1, A_device, 1);
  cublasSetVector(N, sizeof(float), B_host, 1, B_device, 1);

  const float alpha = 3.0;

  // vector addition
  cublasSaxpy(h, N, &alpha, A_device, 1, B_device, 1);

  // transfer device memory to host memory
  cublasGetVector(N, sizeof(float), B_device, 1, C_host, 1);

  // check that we got the correct answer
  check_vector_addition(A_host, B_host, C_host, alpha, N);

  std::cout << "cublas vector addition worked!" << std::endl;

  // free cublas handler
  cublasDestroy(h);

  // free allocated memory
  free(A_host);
  free(B_host);
  cudaFree(A_device);
  cudaFree(B_device);

  return 0;
}
