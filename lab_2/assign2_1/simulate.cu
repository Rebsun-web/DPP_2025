/*
 * simulate.cu
 *
 * Implementation of a wave equation simulation, parallelized on the GPU using
 * CUDA.
 *
 * You are supposed to edit this file with your implementation, and this file
 * only.
 *
 */

#include <cstdlib>
#include <iostream>

#include "simulate.hh"

using namespace std;


/* Utility function, use to do error checking for CUDA calls
 *
 * Use this function like this:
 *     checkCudaCall(<cuda_call>);
 *
 * For example:
 *     checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
 * 
 * Special case to check the result of the last kernel invocation:
 *     kernel<<<...>>>(...);
 *     checkCudaCall(cudaGetLastError());
**/

// Wave equation constant (same as in pthreads version)
#define C 0.15

#define BLOCK_SIZE 256

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void waveKernel(const double* old, const double* current, double* next, int i_max) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= i_max) {
        return;
    }
    
    // Endpoints are 0
    if (i == 0 || i == i_max - 1) {
        next[i] = 0.0;
        return;
    }
    
    // Global memory access
    double left = current[i - 1];
    double center = current[i];
    double right = current[i + 1];
    double old_val = old[i];
    
    next[i] = 2.0 * center - old_val + C * (left - 2.0 * center + right);
}

/* Function that will simulate the wave equation, parallelized using CUDA.
 *
 * i_max: how many data points are on a single wave
 * t_max: how many iterations the simulation should run
 * block_size: how many threads per block you should use
 * old_array: array of size i_max filled with data for t-1
 * current_array: array of size i_max filled with data for t
 * next_array: array of size i_max. You should fill this with t+1
 * 
 */
double *simulate(const long i_max, const long t_max, const long block_size,
                 double *old_array, double *current_array, double *next_array) {
    
    // YOUR CODE HERE
    int threadBlockSize = BLOCK_SIZE;
    int numBlocks = (i_max + threadBlockSize - 1) / threadBlockSize;
    
    // Three arrays on the GPU
    double *deviceOld = NULL;
    double *deviceCurrent = NULL;
    double *deviceNext = NULL;
    
    // Allocate memory
    checkCudaCall(cudaMalloc((void **) &deviceOld, i_max * sizeof(double)));
    checkCudaCall(cudaMalloc((void **) &deviceCurrent, i_max * sizeof(double)));
    checkCudaCall(cudaMalloc((void **) &deviceNext, i_max * sizeof(double)));
    
    // Copy from host to device
    checkCudaCall(cudaMemcpy(deviceOld, old_array, i_max * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceCurrent, current_array, i_max * sizeof(double), cudaMemcpyHostToDevice));

    for (int t = 0; t < t_max; t++) {
        waveKernel<<<numBlocks, threadBlockSize>>>(
            deviceOld, deviceCurrent, deviceNext, i_max
        );
        
        // Check kernel launch errors
        checkCudaCall(cudaGetLastError());
        
        // Pointers rotation
        double *temp = deviceOld;
        deviceOld = deviceCurrent;
        deviceCurrent = deviceNext;
        deviceNext = temp;
    }
    
    checkCudaCall(cudaMemcpy(current_array, deviceCurrent, i_max * sizeof(double), cudaMemcpyDeviceToHost));
    
    checkCudaCall(cudaFree(deviceOld));
    checkCudaCall(cudaFree(deviceCurrent));
    checkCudaCall(cudaFree(deviceNext));

    return current_array;
}

