#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"
#include <sys/time.h>

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = true;
  for (int i = 0; i < N; ++i) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      match = false;
      printf("Arrays do not match!\n");
      printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
      break;
    }
  }
  if (match == true) {
    printf("Arrays match.\n\n");
  }
}

void initialData(float *data, int size) {
  // Generate different seed for random number
  time_t t;
  srand((unsigned int)time(&t));

  for (int i = 0; i < size; ++i) {
    data[i] = (float)(rand() & 0xFF) / 10.0f;
  }
  return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N) {
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
  }
  return;
}

__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) C[i] = A[i] + B[i];
  return;
}

double cpuSecond(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double(tv.tv_sec) + double(tv.tv_usec) * 1.E-6);
}

int main(int argc, char *argv[]) {
  printf("%s Starting...\n", argv[0]);
  // set up device
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, 0));
  printf("Using Device %d: %s\n", 0, deviceProp.name);
  cudaSetDevice(0);
  // set up data size of vectors
  int nElem = 1 << 24;
  printf("Vector size %d\n", nElem);
  // malloc host memory
  size_t nBytes = nElem * sizeof(float);
  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A = (float *)malloc(nBytes);
  h_B = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);

  // initialize data at host side
  initialData(h_A, nElem);
  initialData(h_B, nElem);

  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);

  // malloc device global memory
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, nBytes);
  cudaMalloc(&d_B, nBytes);
  cudaMalloc(&d_C, nBytes);

  // transfer data from host to device
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
  // invoke kernel at host side
  dim3 block(64);
  dim3 grid((nElem + block.x - 1) / block.x);

  double iStart = cpuSecond();
  sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
  CHECK(cudaDeviceSynchronize());
  double iElaps = cpuSecond() - iStart;
  printf("sumArraysOnGPU <<<%d, %d>>> elapsed %f sec.\n", grid.x, block.x, iElaps);

  // copy kernel result back to host side
  cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

  // add vector at host side for result checks
  iStart = cpuSecond();
  sumArraysOnHost(h_A, h_B, hostRef, nElem);
  iElaps = cpuSecond() - iStart;
  printf("sumArraysOnHost elapsed %f sec.\n", iElaps);

  // check device results
  checkResult(hostRef, gpuRef, nElem);

  // free device global memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  // free host memory
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  return 0;
}
