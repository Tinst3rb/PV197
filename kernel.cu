#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define R 128
#define SIZE 1024*1024*32

#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
 ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

unsigned power2_ceil(unsigned x) {
    int power = 2;
    x--;
    while (x >>= 1) power <<= 1;
    return power;
}

__global__ void set_boundary_values_double(double* prefix_sums, const float* input, int size) {
    if (threadIdx.x == 0) {
        prefix_sums[size] = prefix_sums[size-1] + (double)input[size-1];
        prefix_sums[size+1] = (double)input[0];
        prefix_sums[size+2] = (double)input[size-1];
    }
}

__global__ void adjust_block_sums_offset_double(double *out, const double *block_sums_scanned, 
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

template<typename T_out, typename T_in>
__global__ void blelloch_arbitrary_generic(T_out *out, const T_in *in, int n, 
                                           double* block_sums, int ceil_pow_2, int block_offset)
{
    extern __shared__ double temp[];

    int block_index = blockIdx.x + block_offset;
    int local_thread_index = threadIdx.x;
    int block_offset_elements = block_index * ceil_pow_2;
    int offset = 1;

    int global_idx1 = block_offset_elements + 2 * local_thread_index;
    int global_idx2 = block_offset_elements + 2 * local_thread_index + 1;
    
    int ai = 2 * local_thread_index;
    int bi = 2 * local_thread_index + 1;
    int bank_offset_a_i = CONFLICT_FREE_OFFSET(ai);
    int bank_offset_b_i = CONFLICT_FREE_OFFSET(bi);
    
    temp[ai + bank_offset_a_i] = (global_idx1 < n) ? (double)in[global_idx1] : 0.0;
    temp[bi + bank_offset_b_i] = (global_idx2 < n) ? (double)in[global_idx2] : 0.0;

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
        out[global_idx1] = (T_out)temp[ai + bank_offset_a_i];
    }
    if (global_idx2 < n) {
        out[global_idx2] = (T_out)temp[bi + bank_offset_b_i];
    }
}

template<typename T_out, typename T_in>
void blelloch_scan_generic(T_out *out, const T_in *const in, int size)
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
        
        blelloch_arbitrary_generic<T_out, T_in><<<blocks_this_launch, THREADS_PER_BLOCK, shared_mem_size>>>(
            out, in, size, block_sums, ceil_pow_2, start_block);
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("ERROR in blelloch_arbitrary_generic: %s\n", cudaGetErrorString(err));
        }
    }
    
    cudaDeviceSynchronize();

    if (blocks > 1) {
        double *block_sums_scanned;
        cudaMalloc((void **)&block_sums_scanned, blocks * sizeof(double));

        blelloch_scan_generic<double, double>(block_sums_scanned, block_sums, blocks);
        
        for (int start_block = 0; start_block < blocks; start_block += max_blocks_per_launch) {
            int blocks_this_launch = min(max_blocks_per_launch, blocks - start_block);
            
            adjust_block_sums_offset_double<<<blocks_this_launch, THREADS_PER_BLOCK>>>(
                out, block_sums_scanned, size, ceil_pow_2, start_block);
            
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("ERROR in adjust_block_sums_offset_double: %s\n", cudaGetErrorString(err));
            }
        }
        
        cudaFree(block_sums_scanned);
    }

    cudaFree(block_sums);
    cudaDeviceSynchronize();
}

__global__ void sma_double(float* out, double* prefix_sum, int size, int window_size)
{
    extern __shared__ double shared_memory[];
    
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
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error after malloc: %s\n", cudaGetErrorString(err));
        return;
    }
    
    blelloch_scan_generic<double, float>(prefix_sums, input, size); 

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
    
    set_boundary_values_double<<<1, 1>>>(prefix_sums, input, size);
    
    cudaDeviceSynchronize();
    
    int blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int shared_mem_size = ((THREADS_PER_BLOCK + 2 * R) * sizeof(double));

    sma_double<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(average, prefix_sums, size, R);
    
    cudaFree(prefix_sums);
}