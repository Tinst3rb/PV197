#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define R 128
#define SIZE 1024*1024*32

#define THREADS_PER_BLOCK 512
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

unsigned power2_ceil(unsigned x) {
    int power = 2;
    x--;
    while (x >>= 1) power <<= 1;
    return power;
}

__global__ void set_boundary_values(double* prefix_sums, const float* input, int size) {
    if (threadIdx.x == 0) {
        prefix_sums[size] = prefix_sums[size-1] + (double)input[size-1];
        prefix_sums[size+1] = (double)input[0];
        prefix_sums[size+2] = (double)input[size-1];
    }
}

// Adjustment with double precision
__global__ void adjust_block_sums_offset(double *out, const double *block_sums_scanned, 
                                         int n, int ceil_pow_2, int block_offset) 
{
    int block_idx = blockIdx.x + block_offset;
    if (block_idx == 0) return;
    
    double adjustment = block_sums_scanned[block_idx];
    int block_start = block_idx * ceil_pow_2;
    int block_end = min(block_start + ceil_pow_2, n);
    
    for (int idx = block_start + threadIdx.x; idx < block_end; idx += blockDim.x) {
        out[idx] += adjustment;
    }
}

// Fast float scan, outputs double
__global__ void blelloch_arbitrary(double *out, const float *in, int n, 
                                   double* block_sums, int ceil_pow_2, int block_offset)
{
    extern __shared__ char shared_mem[];  // Use char array
    float* temp = (float*)shared_mem;      // Cast to float

    int block_index = blockIdx.x + block_offset;
    int local_thread_index = threadIdx.x;
    int block_offset_elements = block_index * ceil_pow_2;
    int offset = 1;

    int global_idx1 = block_offset_elements + 2 * local_thread_index;
    int global_idx2 = block_offset_elements + 2 * local_thread_index + 1;
    
    int ai = 2 * local_thread_index;
    int bi = 2 * local_thread_index + 1;
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
    temp[ai + bankOffsetA] = (global_idx1 < n) ? in[global_idx1] : 0.0f;
    temp[bi + bankOffsetB] = (global_idx2 < n) ? in[global_idx2] : 0.0f;

    for (int d = ceil_pow_2 >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (local_thread_index < d)
        {
            int a_i = offset * (2 * local_thread_index + 1) - 1;
            int b_i = offset * (2 * local_thread_index + 2) - 1;
            a_i += CONFLICT_FREE_OFFSET(a_i);
            b_i += CONFLICT_FREE_OFFSET(b_i);
            temp[b_i] += temp[a_i];
        }
        offset *= 2;
    }

    if (local_thread_index == 0)
    {
        int last_idx = ceil_pow_2 - 1;
        last_idx += CONFLICT_FREE_OFFSET(last_idx);
        if (block_sums != nullptr) {
            block_sums[block_index] = (double)temp[last_idx];
        }
        temp[last_idx] = 0.0f;
    }
    __syncthreads();

    for (int d = 1; d < ceil_pow_2; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (local_thread_index < d)
        {
            int a_i = offset * (2 * local_thread_index + 1) - 1;
            int b_i = offset * (2 * local_thread_index + 2) - 1;
            a_i += CONFLICT_FREE_OFFSET(a_i);
            b_i += CONFLICT_FREE_OFFSET(b_i);
            float t = temp[a_i];
            temp[a_i] = temp[b_i];
            temp[b_i] += t;
        }
    }

    __syncthreads();

    if (global_idx1 < n) {
        out[global_idx1] = (double)temp[ai + bankOffsetA];
    }
    if (global_idx2 < n) {
        out[global_idx2] = (double)temp[bi + bankOffsetB];
    }
}

// Double->double scan for recursion
__global__ void blelloch_arbitrary_double(double *out, const double *in, int n, 
                                          double* block_sums, int ceil_pow_2, int block_offset)
{
    extern __shared__ char shared_mem[];  // Use char array
    double* temp = (double*)shared_mem;    // Cast to double

    int block_index = blockIdx.x + block_offset;
    int local_thread_index = threadIdx.x;
    int block_offset_elements = block_index * ceil_pow_2;
    int offset = 1;

    int global_idx1 = block_offset_elements + 2 * local_thread_index;
    int global_idx2 = block_offset_elements + 2 * local_thread_index + 1;
    
    int ai = 2 * local_thread_index;
    int bi = 2 * local_thread_index + 1;
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    
    temp[ai + bankOffsetA] = (global_idx1 < n) ? in[global_idx1] : 0.0;
    temp[bi + bankOffsetB] = (global_idx2 < n) ? in[global_idx2] : 0.0;

    for (int d = ceil_pow_2 >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (local_thread_index < d)
        {
            int a_i = offset * (2 * local_thread_index + 1) - 1;
            int b_i = offset * (2 * local_thread_index + 2) - 1;
            a_i += CONFLICT_FREE_OFFSET(a_i);
            b_i += CONFLICT_FREE_OFFSET(b_i);
            temp[b_i] += temp[a_i];
        }
        offset *= 2;
    }

    if (local_thread_index == 0)
    {
        int last_idx = ceil_pow_2 - 1;
        last_idx += CONFLICT_FREE_OFFSET(last_idx);
        if (block_sums != nullptr) {
            block_sums[block_index] = temp[last_idx];
        }
        temp[last_idx] = 0.0;
    }
    __syncthreads();

    for (int d = 1; d < ceil_pow_2; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (local_thread_index < d)
        {
            int a_i = offset * (2 * local_thread_index + 1) - 1;
            int b_i = offset * (2 * local_thread_index + 2) - 1;
            a_i += CONFLICT_FREE_OFFSET(a_i);
            b_i += CONFLICT_FREE_OFFSET(b_i);
            double t = temp[a_i];
            temp[a_i] = temp[b_i];
            temp[b_i] += t;
        }
    }

    __syncthreads();

    if (global_idx1 < n) {
        out[global_idx1] = temp[ai + bankOffsetA];
    }
    if (global_idx2 < n) {
        out[global_idx2] = temp[bi + bankOffsetB];
    }
}

void blelloch_scan_arbitrary_array_double(double *out, const double *const in, int size);

void blelloch_scan_arbitrary_array(double *out, const float *const in, int size)
{        
    int ceil_pow_2 = power2_ceil(ELEMENTS_PER_BLOCK);
    int blocks = (size + ceil_pow_2 - 1) / ceil_pow_2;
    
    double *block_sums;
    cudaMalloc((void **)&block_sums, blocks * sizeof(double));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_blocks_per_launch = prop.maxGridSize[0];
    
    // Float shared memory for speed!
    int shared_mem_size = (2 * ceil_pow_2 + CONFLICT_FREE_OFFSET(2 * ceil_pow_2 - 1)) * sizeof(float);
    
    for (int start_block = 0; start_block < blocks; start_block += max_blocks_per_launch) {
        int blocks_this_launch = min(max_blocks_per_launch, blocks - start_block);
        
        blelloch_arbitrary<<<blocks_this_launch, THREADS_PER_BLOCK, shared_mem_size>>>(
            out, in, size, block_sums, ceil_pow_2, start_block);
    }    

    if (blocks > 1) {
        double *block_sums_scanned;
        cudaMalloc((void **)&block_sums_scanned, blocks * sizeof(double));

        blelloch_scan_arbitrary_array_double(block_sums_scanned, block_sums, blocks);
        
        for (int start_block = 0; start_block < blocks; start_block += max_blocks_per_launch) {
            int blocks_this_launch = min(max_blocks_per_launch, blocks - start_block);
            
            adjust_block_sums_offset<<<blocks_this_launch, THREADS_PER_BLOCK>>>(
                out, block_sums_scanned, size, ceil_pow_2, start_block);
        }
        
        cudaFree(block_sums_scanned);
    }

    cudaFree(block_sums);
}

void blelloch_scan_arbitrary_array_double(double *out, const double *const in, int size)
{        
    int ceil_pow_2 = power2_ceil(ELEMENTS_PER_BLOCK);
    int blocks = (size + ceil_pow_2 - 1) / ceil_pow_2;
    
    double *block_sums;
    cudaMalloc((void **)&block_sums, blocks * sizeof(double));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int max_blocks_per_launch = prop.maxGridSize[0];
    
    int shared_mem_size = (2 * ceil_pow_2 + CONFLICT_FREE_OFFSET(2 * ceil_pow_2 - 1)) * sizeof(double);
    
    for (int start_block = 0; start_block < blocks; start_block += max_blocks_per_launch) {
        int blocks_this_launch = min(max_blocks_per_launch, blocks - start_block);
        
        blelloch_arbitrary_double<<<blocks_this_launch, THREADS_PER_BLOCK, shared_mem_size>>>(
            out, in, size, block_sums, ceil_pow_2, start_block);
    }
    
    if (blocks > 1) {
        double *block_sums_scanned;
        cudaMalloc((void **)&block_sums_scanned, blocks * sizeof(double));

        blelloch_scan_arbitrary_array_double(block_sums_scanned, block_sums, blocks);
        
        for (int start_block = 0; start_block < blocks; start_block += max_blocks_per_launch) {
            int blocks_this_launch = min(max_blocks_per_launch, blocks - start_block);
            
            adjust_block_sums_offset<<<blocks_this_launch, THREADS_PER_BLOCK>>>(
                out, block_sums_scanned, size, ceil_pow_2, start_block);
        }
        
        cudaFree(block_sums_scanned);
    }

    cudaFree(block_sums);
}

__global__ void sma(float* out, double* prefix_sum, int size, int window_size)
{
    extern __shared__ char shared_mem[];   // Use char array
    double* shared_memory = (double*)shared_mem;  // Cast to double
    
    int block_start = blockIdx.x * blockDim.x;
    int global_thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    
    int load_size = blockDim.x + 2 * window_size;
    int load_start = block_start - window_size;
    
    for (int i = threadIdx.x; i < load_size; i += blockDim.x) {
        int global_load_index = load_start + i;
        
        if (global_load_index < 0) {
            shared_memory[i] = (double)global_load_index * prefix_sum[size+1];
        } 
        else if (global_load_index >= size) {
            shared_memory[i] = prefix_sum[size] + (double)(global_load_index - size) * prefix_sum[size+2];
        } 
        else {
            shared_memory[i] = prefix_sum[global_load_index];
        }
    }
    
    __syncthreads();

    if (global_thread_index < size) {
        int left_idx = threadIdx.x;
        int right_idx = threadIdx.x + 2 * window_size;
        double left_val = shared_memory[left_idx];
        double right_val = shared_memory[right_idx];
        double result = (right_val - left_val) / (2.0 * window_size);
        out[global_thread_index] = (float)result;
    }
}

void solveGPU(float *average, const float *const input, int size)
{
    double *prefix_sums;
    cudaMalloc((void **)&prefix_sums, (size + 3) * sizeof(double));
    
    blelloch_scan_arbitrary_array(prefix_sums, input, size); 
    
    set_boundary_values<<<1, 1>>>(prefix_sums, input, size);
    
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int shared_mem_size = ((THREADS_PER_BLOCK + 2 * R) * sizeof(double));

    sma<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(average, prefix_sums, size, R);
    
    cudaFree(prefix_sums);
}