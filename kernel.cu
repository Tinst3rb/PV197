#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define R 256
#define SIZE 1024*1024*32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 12

__global__ void compute_averages(float *out, const float *in, int size) {
    extern __shared__ float tile[];
    
    const int tid = threadIdx.x;
    const int blockStart = blockIdx.x * BLOCK_SIZE * ELEMENTS_PER_THREAD;
    const int tileStart = blockStart - R;
    const int tileSize = BLOCK_SIZE * ELEMENTS_PER_THREAD + 2 * R;
    
    for (int i = tid; i < tileSize; i += BLOCK_SIZE) {
        int globalIdx = tileStart + i;
        globalIdx = (globalIdx < 0) ? 0 : ((globalIdx >= size) ? size - 1 : globalIdx);
        tile[i] = in[globalIdx];
    }
    
    __syncthreads();
    
    int gid = blockStart + tid * ELEMENTS_PER_THREAD;
    if (gid >= size) return;
    
    const int baseLocalIdx = tid * ELEMENTS_PER_THREAD + R;
    const float invWindow = 1.0f / 512.0f;
    
    float a0=0, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0, a7=0;
    
    #pragma unroll 64
    for (int w = 0; w < 512; w += 8) {
        int i = baseLocalIdx - R + w;
        a0 += tile[i];   a1 += tile[i+1];
        a2 += tile[i+2]; a3 += tile[i+3];
        a4 += tile[i+4]; a5 += tile[i+5];
        a6 += tile[i+6]; a7 += tile[i+7];
    }
    
    float sum = ((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7));
    out[gid++] = sum * invWindow;
    
    #pragma unroll
    for (int e = 1; e < ELEMENTS_PER_THREAD; e++) {
        if (gid >= size) return;
        sum += tile[baseLocalIdx+e+R-1] - tile[baseLocalIdx+e-R-1];
        out[gid++] = sum * invWindow;
    }
}

void solveGPU(float* average, const float* const input, int size) {
    const int numBlocks = (size + BLOCK_SIZE * ELEMENTS_PER_THREAD - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD);
    const int sharedMemSize = (BLOCK_SIZE * ELEMENTS_PER_THREAD + 2 * R) * sizeof(float);
    
    compute_averages<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(average, input, size);
}