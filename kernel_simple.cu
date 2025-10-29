#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define R 256

__global__ void movingAverageKernel(float* out, const float* in, int size) {
    extern __shared__ float shared[];
    
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;
    
    int sharedSize = blockDim.x + 2 * R;
    int blockStart = blockIdx.x * blockDim.x - R;
    
    for (int i = localIdx; i < sharedSize; i += blockDim.x) {
        int globalPos = blockStart + i;
        
        if (globalPos < 0) {
            shared[i] = in[0];
        } else if (globalPos >= size) {
            shared[i] = in[size - 1];
        } else {
            shared[i] = in[globalPos];
        }
    }
    
    __syncthreads();
    
    if (localIdx == 0) {
        for (int i = 1; i < sharedSize; i++) {
            shared[i] += shared[i - 1];
        }
    }
    
    __syncthreads();
    
    if (globalIdx < size) {
        int sharedStart = localIdx;
        int leftIdx = sharedStart - 1;
        int rightIdx = sharedStart + 2 * R - 1;
        
        float windowSum;
        if (leftIdx < 0) {
            windowSum = shared[rightIdx];
        } else {
            windowSum = shared[rightIdx] - shared[leftIdx];
        }
        
        out[globalIdx] = windowSum / (2.0f * R);
    }
}

void solveGPU(float* average, const float* const input, int size) {
    int blockSize = 128;
    int numBlocks = (size + blockSize - 1) / blockSize;
    int sharedMemSize = (blockSize + 2 * R) * sizeof(float);
    
    movingAverageKernel<<<numBlocks, blockSize, sharedMemSize>>>(average, input, size);
    cudaDeviceSynchronize();
}