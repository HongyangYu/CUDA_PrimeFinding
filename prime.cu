#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

typedef unsigned long long int uint64_t;

/******************************************************************************
* kernel for finding prime numbers using the sieve of eratosthenes
* - primes: an array of bools. initially all numbers are set to "0".
*			  A "0" value means that the number at that index is prime.
* - max: the max size of the primes array
******************************************************************************/
__global__ static void basicPrime(char *primes, uint64_t max, int base, int start) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index == 0 || index == 1) {
		primes[0] = 1;
		primes[1] = 1;
		return;
	}
	if (index >= start && primes[index] == 0 && index % base == 0){
		primes[index] = 1;
	}
}
 __global__ static void segmentSieve(char *primes, uint64_t max, uint64_t maxRoot) {
	int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (index > 0) {
		int low = index * maxRoot;
		if (low > max) return;
		int high = low + maxRoot;
		if (high > max) high = max;
		for (int i = 0; i < maxRoot; i++) {
			if (primes[i] == 0) {
				int loLim = (low / i) * i;
				if (loLim < low) {
					loLim += i;
				}
				for (int j = loLim; j < high; j += i) {
					primes[j] = 1;
				}
			}
		}
	}
}
/******************************************************************************
* checkDevice()
******************************************************************************/
__host__ int checkDevice() {
	// query the Device and decide on the block size
	int devID = 0; // the default device ID
	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);
	if (error != cudaSuccess){
		printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
		exit(EXIT_FAILURE);
	}
	error = cudaGetDeviceProperties(&deviceProp, devID);
	if (deviceProp.computeMode == cudaComputeModeProhibited || error != cudaSuccess){
		printf("CUDA device ComputeMode is prohibited or failed to getDeviceProperties\n");
		return EXIT_FAILURE;
	}
	// Use a larger block size for Fermi and above (see compute capability)
	return (deviceProp.major < 2) ? 16 : 32;
}

/******************************************************************************
* genPrimesOnDevice
* - inputs: limit - the largest prime that should be computed
*			primes - an array of size [limit], initialized to 0
******************************************************************************/
__host__ void genPrimesOnDevice(char* primes, uint64_t max){
	int blockSize = checkDevice();
	if (blockSize == EXIT_FAILURE)
		return;
	char* d_Primes = NULL;
	int sizePrimes = sizeof(char) * max;
	uint64_t maxRoot = sqrt(max);
	maxRoot = (max+maxRoot-1)/maxRoot;//celling
	// allocate the primes on the device and set them to 0
	cudaMalloc(&d_Primes, sizePrimes);
	cudaMemset(d_Primes, 0, sizePrimes);
	// make sure that there are no errors...
	cudaPeekAtLastError();
	// setup the execution configuration
	dim3 dimBlock(maxRoot, 1, 1); 
	dim3 dimGrid(1);
	//////// debug
	#ifdef DEBUG
	printf("dimBlock(%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
	printf("dimGrid(%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
	#endif
	// call the kernel
	for (int i = 2; i * i < maxRoot; i++) {
		basicPrime<<<dimGrid, dimBlock1>>>(d_Primes, maxRoot, i, i * i);
	}
	segmentSieve<<<dimGrid, dimBlock2>>>(d_Primes, max, maxRoot);
	// check for kernel errors
	cudaPeekAtLastError();
	cudaDeviceSynchronize();
	// copy the results back
	cudaMemcpy(primes, d_Primes, sizePrimes, cudaMemcpyDeviceToHost);
	// no memory leaks
	cudaFree(d_Primes);
}
/******************************************************************************
******************************************************************************/
int main() {
	uint64_t maxPrime = 1<<20; // find all primes from 0-101
	char* primes = (char*) malloc(maxPrime);
	memset(primes, 0, maxPrime); // initialize all elements to 0
	genPrimesOnDevice(primes, maxPrime);
	// display the results
	int i,sum = 0;
	for (i = 0; i < maxPrime; i++) {
		if (primes[i] == 0) {
			// printf("%i ", i);
			sum += 1;
		}
	}
	printf("number of Prime: %i ", sum);
	printf("\n");
	free(primes);
	return 0;
}