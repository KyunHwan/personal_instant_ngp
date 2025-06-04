#include <iostream>
#include <utility>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

__global__ void matrixMulKernel(const float* M, const float* N, float* P, int Width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float Pval = 0;
  for (int i = 0; i < Width; i++) {
    Pval += M[row * Width + i] * N[col + i * Width];
  }
  P[row * Width + col] = Pval;
}

__global__ void matrixTileMulKernel(const float* M,
                                const float* N,
                                float* P,
                                int Width)
{
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  // Block and thread indices
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Compute the row and column of P that this thread will compute
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  // Accumulator for the dot‐product
  float Pvalue = 0.0f;

  // Loop over all tiles (phases)
  int numTiles = Width / TILE_WIDTH;
  for (int ph = 0; ph < numTiles; ++ph) {
    // Load one element of M into shared memory:
    //   Global element: M[Row, ph*TILE_WIDTH + tx]
    Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];

    // Load one element of N into shared memory:
    //   Global element: N[ph*TILE_WIDTH + ty, Col]
    Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];

    // Wait until all threads have written their piece of Mds and Nds
    __syncthreads();

    // Perform the partial dot‐product for this tile
    for (int k = 0; k < TILE_WIDTH; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }

    // Wait before loading the next tile to avoid race conditions
    __syncthreads();
  }

  // Write the final result to global memory
  P[Row * Width + Col] = Pvalue;
}

__global__ void device_vector_add(const float *a, const float *b, float *c, int n) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void device_global_mem_vector_scale(const float* A, float* B, float k, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    B[i] = k * A[i];
  }
}

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);

    }
}

template <typename F, typename... Args>
auto cudaExecutionTimer(F&& f, Args&&... args)
    -> decltype(std::forward<F>(f)(std::forward<Args>(args)...))
{
  cudaEvent_t start, stop;
  cudaCheckError(cudaEventCreate(&start));
  cudaCheckError(cudaEventCreate(&stop));

  cudaCheckError(cudaEventRecord(start, 0));

  std::forward<F>(f)(std::forward<Args>(args)...);

  cudaCheckError(cudaEventRecord(stop, 0));

  cudaCheckError(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Elapsed GPU time (matrixMulKernel): %f ms\n", milliseconds);

  cudaCheckError(cudaEventDestroy(start));
  cudaCheckError(cudaEventDestroy(stop));
  return std::forward<F>(f)(std::forward<Args>(args)...);
}

void host_vector_add() {
  int N = 1024;
  size_t size = N * sizeof(float);
  float* host_a = (float*)malloc(size);
  float* host_b = (float*)malloc(size);
  float* host_c = (float*)malloc(size);

  for (int i = 0; i < N; ++i) {
    host_a[i] = static_cast<float>(i);
    host_b[i] = static_cast<float>(2 * i);
  }

  float *device_a, *device_b, *device_c;
  cudaCheckError(cudaMalloc((void**)&device_a, size));
  cudaCheckError(cudaMalloc((void**)&device_b, size));
  cudaCheckError(cudaMalloc((void**)&device_c, size));

  cudaCheckError(cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  device_vector_add<<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, device_c, N);
  cudaCheckError(cudaDeviceSynchronize());

  cudaCheckError(cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost));

  for (int i = 0; i < 10; ++i) {
    std::cout << "host_c[" << i << "] = " << host_c[i] << std::endl;
  }

  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
  free(host_a);
  free(host_b);
  free(host_c);
}

void host_vector_scale() {
  int N = 1 << 20; // 1 million elements
  float k = 2.5f;

  size_t size = N * sizeof(float);
  float *host_A = (float*)malloc(size);
  float *host_B = (float*)malloc(size);

  // Initialize input
  for (int i = 0; i < N; ++i) host_A[i] = static_cast<float>(i);

  // Allocate device memory
  float *device_A, *device_B;
  cudaCheckError(cudaMalloc((void**)&device_A, size));
  cudaCheckError(cudaMalloc((void**)&device_B, size));

  // Copy to device
  cudaCheckError(cudaMemcpy(device_A, host_A, size, cudaMemcpyHostToDevice));

  // Launch
  int threadsPerBlock = 256;
  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  device_global_mem_vector_scale<<<blocks, threadsPerBlock>>>(device_A, device_B, k, N);

  // Copy back
  cudaCheckError(cudaMemcpy(host_B, device_B, size, cudaMemcpyDeviceToHost));

  // Validate a few values
  for (int i = 0; i < 5; ++i)
    std::cout << host_B[i] << " ";
  std::cout << std::endl;

  cudaFree(device_A);
  cudaFree(device_B);
  free(host_A);
  free(host_B);
}

void matrixMultiply()
{
  // Choose matrix dimension (must be a multiple of TILE_WIDTH)
  int Width = 8192;  // e.g., 1024 = 16 * 64
  if (Width % TILE_WIDTH != 0) {
    fprintf(stderr, "Error: Width must be a multiple of TILE_WIDTH (%d)\n", TILE_WIDTH);
    exit(EXIT_FAILURE);
  }

  size_t numElements = (size_t)Width * (size_t)Width;
  size_t bytes = numElements * sizeof(float);

  // Allocate host memory
  float* host_M = (float*)malloc(bytes);
  float* host_N = (float*)malloc(bytes);
  float* host_P = (float*)malloc(bytes);

  if (!host_M || !host_N || !host_P) {
    fprintf(stderr, "Failed to allocate host matrices\n");
    exit(EXIT_FAILURE);
  }

  // Initialize input matrices host_M and host_N
  for (int row = 0; row < Width; ++row) {
    for (int col = 0; col < Width; ++col) {
      host_M[row * Width + col] = static_cast<float>(row + col);
      host_N[row * Width + col] = static_cast<float>(row - col);
    }
  }

  float *device_M = nullptr, *device_N = nullptr, *device_P = nullptr;

  cudaCheckError(cudaMalloc(&device_M, bytes));
  cudaCheckError(cudaMalloc(&device_N, bytes));
  cudaCheckError(cudaMalloc(&device_P, bytes));

  cudaCheckError(cudaMemcpy(device_M, host_M, bytes, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(device_N, host_N, bytes, cudaMemcpyHostToDevice));

  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);


  auto kernelLauncher = [&]() {
    matrixTileMulKernel<<<dimGrid, dimBlock>>>(device_M, device_N, device_P, Width);
    //matrixMulKernel<<<dimGrid, dimBlock>>>(device_M, device_N, device_P, Width);
    cudaCheckError(cudaGetLastError());
  };

  cudaExecutionTimer(kernelLauncher);

  cudaCheckError(cudaMemcpy(host_P, device_P, bytes, cudaMemcpyDeviceToHost));

  cudaCheckError(cudaFree(device_M));
  cudaCheckError(cudaFree(device_N));
  cudaCheckError(cudaFree(device_P));

  free(host_M);
  free(host_N);
  free(host_P);

}


int main() {

  matrixMultiply();
  return 0;
}