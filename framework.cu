#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "kernel.cu"
#include "kernel_CPU.C"

int main(int argc, char **argv){
    // CPU data
    float *input, *average, *average_gpu;
    input = average = NULL;
    // GPU counterparts
    float *dinput, *daverage;
    dinput = daverage = NULL;

    // parse command line
    int device = 0;
    if (argc == 2) 
        device = atoi(argv[1]);
    if (cudaSetDevice(device) != cudaSuccess){
        fprintf(stderr, "Cannot set CUDA device!\n");
        exit(1);
    }
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Using device %d: \"%s\"\n", device, deviceProp.name);

    // create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // allocate and set host memory
    input = (float*)malloc(SIZE*sizeof(input[0]));
    average = (float*)malloc(SIZE*sizeof(average[0]));
    average_gpu = (float*)malloc(SIZE*sizeof(average[0]));
    for (int i = 0; i < SIZE; i++)
        input[i] = (float)rand() / float(RAND_MAX);
 
    // allocate and set device memory
    if (cudaMalloc((void**)&dinput, SIZE*sizeof(dinput[0])) != cudaSuccess
    || cudaMalloc((void**)&daverage, SIZE*sizeof(daverage[0])) != cudaSuccess){
        fprintf(stderr, "Device memory allocation error!\n");
        goto cleanup;
    }
    cudaMemcpy(dinput, input, SIZE*sizeof(dinput[0]), cudaMemcpyHostToDevice);

    // solve on CPU
    printf("Solving on CPU...\n");
    cudaEventRecord(start, 0);
    solveCPU(average, input, SIZE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("CPU performance: %f megavalues/s\n",
        float(SIZE)/time/1e3f);

    // solve on GPU
    printf("Solving on GPU...\n");
    solveGPU(daverage, dinput, SIZE);
    cudaEventRecord(start, 0);
    // for(int i =0; i < 100; i++) 
        solveGPU(daverage, dinput, SIZE);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("GPU performance: %f megavalues/s\n",
        float(SIZE)/time/1e3f);

    // check GPU results
    cudaMemcpy(average_gpu, daverage, SIZE*sizeof(average_gpu[0]), cudaMemcpyDeviceToHost);
    for (int i = 0; i < SIZE; i++)
        if ((average_gpu[i] != average_gpu[i]) /* catches NaN */
	|| (std::abs(average[i]-average_gpu[i]) > 0.0001f)) { 
            fprintf(stderr, "Data mismatch at index %i: %f should be %f :-(\n", i, average_gpu[i], average[i]);
            goto cleanup;
        }
    printf("Test OK.\n");

cleanup:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (daverage) cudaFree(daverage);
    if (dinput) cudaFree(dinput);
    if (average) free(average);
    if (input) free(input);
    if (average_gpu) free(average_gpu);

    return 0;
}
