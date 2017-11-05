#include <cuda_runtime.h>
#include <stdio.h>
#include "common.h"

void initialInt(int *data, const int size) {
  for (int i = 0; i < size; ++i) {
    data[i] = i;
  }
  return;
}

void printMatrix(int *mat, const int nx, const int ny) {
  printf("Matrix (%d x %d):\n", ny, nx);
  for (int iy = 0; iy < ny; ++iy) {
    for (int ix = 0; ix < nx; ++ix) {
      printf("%3d", mat[iy * nx + ix]);
    }
    printf("\n");
  }
  printf("\n");
  return;
}

__global__ void printThreadIndex(int *mat, const int nx, const int ny) {
  int ix = blockIdx.x * blockDim.x + threadIdx.x;
  int iy = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned int idx = iy * nx + ix;
  printf("ThreadIdx(%d, %d), BlockIdx(%d, %d), coordinate(%d, %d), global index %d ival %d\n",
         threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, mat[idx]);
  return;
}

int main(int argc, char *argv[]) {
  printf("%s Starting...\n", argv[0]);
  // set up device
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, 0));
  printf("Using Device %d: %s\n", 0, deviceProp.name);
  cudaSetDevice(0);
  // set up data size of vectors
  int nx = 8;
  int ny = 6;
  int nElem = nx * ny;
  // malloc host memory
  size_t nBytes = nElem * sizeof(int);
  int *h_A = (int *)malloc(nBytes);

  // initialize data at host side
  initialInt(h_A, nElem);

  // malloc device global memory
  int *d_A;
  cudaMalloc(&d_A, nBytes);

  // transfer data from host to device
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  // invoke kernel at host side
  dim3 block(4, 2);
  dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

  printThreadIndex<<<grid, block>>>(d_A, nx, ny);
  CHECK(cudaDeviceSynchronize());

  // free device global memory
  cudaFree(d_A);

  // free host memory
  free(h_A);

  CHECK(cudaDeviceReset());
  return 0;
}
