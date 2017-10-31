#include <stdio.h>
#include "common.h"

__global__ void helloFromGPU(void) {
  if (threadIdx.x == 5) {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
  }
}

int main(int argc, char **argv) {
  printf("Hello World from CPU!\n");

  helloFromGPU<<<1, 10>>>();
  CHECK(cudaDeviceReset());
  return 0;
}
