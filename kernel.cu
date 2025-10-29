#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define R 32
#define SIZE 1024*1024*32

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)


unsigned power2_ceil(unsigned x) {
    int power = 2;
    x--;
    while (x >>= 1) power <<= 1;
    return power;
}


__global__ void set_boundary_values(float* prefix_sums, const float* input, int size) {
    if (threadIdx.x == 0) {
        prefix_sums[size] = prefix_sums[size-1] + input[size-1];
        prefix_sums[size+1] = input[0];
        prefix_sums[size+2] = input[size-1];
    }
}


__global__ void adjust_block_sums_offset(float *out, const float *block_sums_scanned, 
                                         int n, int ceil_pow_2, int block_offset) 
{
    int block_idx = blockIdx.x + block_offset;
    
    if (block_idx == 0) return;
    
    float adjustment = block_sums_scanned[block_idx];
    int block_start = block_idx * ceil_pow_2;
    int block_end = min(block_start + ceil_pow_2, n);
    
    for (int idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        out[idx] += adjustment;
    }
    
    __syncthreads();

}


__global__ void blelloch_arbitrary(float *out, const float *in, int n, 
                                   float* block_sums, int ceil_pow_2, int block_offset)
{
    extern __shared__ float temp[];

    int block_index = blockIdx.x + block_offset;
    int local_thread_index = threadIdx.x;
    int block_offset_elements = block_index * ceil_pow_2;
    int offset = 1;

    // Load input with proper bounds checking
    int global_idx1 = block_offset_elements + 2 * local_thread_index;
    int global_idx2 = block_offset_elements + 2 * local_thread_index + 1;
    
    temp[2 * local_thread_index] = (global_idx1 < n) ? in[global_idx1] : 0;
    temp[2 * local_thread_index + 1] = (global_idx2 < n) ? in[global_idx2] : 0;

    // Up-sweep
    for (int d = ceil_pow_2 >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (local_thread_index < d)
        {
            int a_i = offset * (2 * local_thread_index + 1) - 1;
            int b_i = offset * (2 * local_thread_index + 2) - 1;
            temp[b_i] += temp[a_i];
        }

        offset *= 2;
    }

    // Save block sum and zero last element
    if (local_thread_index == 0)
    {
        
        if (block_sums != nullptr) {
            block_sums[block_index] = temp[ceil_pow_2 - 1];
        }
        temp[ceil_pow_2 - 1] = 0;
    }
    __syncthreads();

    // Down-sweep
    for (int d = 1; d < ceil_pow_2; d *= 2)
    {
        offset >>= 1;
        __syncthreads();

        if (local_thread_index < d)
        {
            int a_i = offset * (2 * local_thread_index + 1) - 1;
            int b_i = offset * (2 * local_thread_index + 2) - 1;
            float t = temp[a_i];
            temp[a_i] = temp[b_i];
            temp[b_i] += t;
        }
    }

    __syncthreads();

    if (global_idx1 < n) {
        out[global_idx1] = temp[2 * local_thread_index];
    }
    if (global_idx2 < n) {
        out[global_idx2] = temp[2 * local_thread_index + 1];
    }
    
}


void blelloch_scan_arbitrary_array(float *out, const float *const in, int size)
{        
    int ceil_pow_2 = power2_ceil(ELEMENTS_PER_BLOCK);
    int blocks = (size + ceil_pow_2 - 1) / ceil_pow_2;
    
    float *block_sums;
    cudaMalloc((void **)&block_sums, blocks * sizeof(float));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_blocks_per_launch = prop.maxGridSize[0];
    
    for (int start_block = 0; start_block < blocks; start_block += max_blocks_per_launch) {
        int blocks_this_launch = min(max_blocks_per_launch, blocks - start_block);
        
        blelloch_arbitrary<<<blocks_this_launch, THREADS_PER_BLOCK, 2 * ceil_pow_2 * sizeof(float)>>>(
            out, in, size, block_sums, ceil_pow_2, start_block);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR in blelloch_arbitrary: %s\n", cudaGetErrorString(err));
        }
    }
    
    cudaDeviceSynchronize();

    if (blocks > 1) {
        float *block_sums_scanned;
        cudaMalloc((void **)&block_sums_scanned, blocks * sizeof(float));

        blelloch_scan_arbitrary_array(block_sums_scanned, block_sums, blocks);
        
        for (int start_block = 0; start_block < blocks; start_block += max_blocks_per_launch) {
            int blocks_this_launch = min(max_blocks_per_launch, blocks - start_block);
            
            adjust_block_sums_offset<<<blocks_this_launch, THREADS_PER_BLOCK>>>(
                out, block_sums_scanned, size, ceil_pow_2, start_block);
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("ERROR in adjust_block_sums_offset: %s\n", cudaGetErrorString(err));
            }
        }
        
        cudaFree(block_sums_scanned);
    }

    cudaFree(block_sums);
    
    cudaDeviceSynchronize();
}


__global__ void sma(float* out, float* prefix_sum, int size, int window_size)
{
    extern __shared__ float shared_memory[];
    
    int block_start = blockIdx.x * blockDim.x;
    int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int load_size = blockDim.x + 2 * window_size;
    int load_start = block_start - window_size;
    
    // Load prefix sum values into shared memory
    for (int i = threadIdx.x; i < load_size; i += blockDim.x) {
        int global_load_index = load_start + i;
        
        if (global_load_index < 0) {
            shared_memory[i] = (float)global_load_index * prefix_sum[size+1];
        } 
        else if (global_load_index > size) {
            shared_memory[i] = prefix_sum[size] + (float)(global_load_index - size) * prefix_sum[size+2];
        } 
        else {
            shared_memory[i] = prefix_sum[global_load_index];
        }
    }
    
    __syncthreads();

    if (global_thread_index < size) {
        int left_idx = threadIdx.x;
        int right_idx = threadIdx.x + 2 * window_size;
        float left_val = shared_memory[left_idx];
        float right_val = shared_memory[right_idx];
        float result = (right_val - left_val) / (2.0f * window_size);
        out[global_thread_index] = result;
    }
}


void solveGPU(float *average, const float *const input, int size)
{
    float *prefix_sums;
    cudaMalloc((void **)&prefix_sums, (size + 3) * sizeof(float));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after malloc: %s\n", cudaGetErrorString(err));
        return;
    }
    
    blelloch_scan_arbitrary_array(prefix_sums, input, size); 

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
        return;
    }

    set_boundary_values<<<1, 1>>>(prefix_sums, input, size);
    
    cudaDeviceSynchronize();
    
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int shared_mem_size = ((THREADS_PER_BLOCK + 2 * R) * sizeof(float));

    sma<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(average, prefix_sums, size, R);

    // Copy results back to host for printing
    float *h_aver = (float*)malloc(size * sizeof(float));
    cudaMemcpy(h_aver, average, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print prefix sums
    printf("averages:\n");
    for (int i = size - 100; i < size; i+=1) {
        printf("average[%d] = %f\n", i, h_aver[i]);
    }
    printf("\n");
    
    // Free host memory
    free(h_aver);
    
    cudaFree(prefix_sums);
}
