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

__global__ static void sieveOfEratosthenes(char *primes, uint64_t max) {
	primes[0] = 1; // value of 1 means the number is NOT prime
	primes[1] = 1; // numbers "0" and "1" are not prime numbers
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t maxRoot = sqrt((double)max);
	// make sure index won't go out of bounds, 
	// also don't execute it on index 1
	if (index <= maxRoot && primes[index] == 0 && index > 1 ){
		// mark off the composite numbers
		for (int j = index * index; j < max; j += index){
			primes[j] = 1;
		}
		
	}
}

__global__ static void segmentSieve(char *primes, uint64_t max) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if(index>0){
		const uint64_t maxRoot = sqrt((double)max);
		int low = maxRoot*index;
		int high = low + maxRoot;
		if(high > max) high = max;
		for (int i = 2; i < maxRoot; i++){ //sqrt(n)lglg(sqrt(n))
			if(primes[i]==0){
				int loLim = (low / i) * i;
				if (loLim < low)
					loLim += i;
				for (int j=loLim; j<high; j+=i) 
					primes[j] = 1;
			}
		
		} 
	}
}

/*******************************************************************************
 * checkDevice()
 ******************************************************************************/
__host__ int checkDevice(){
	printf("Checking device...\n");
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
	uint64_t maxRoot = sqrt(max)+1;
	uint64_t maxRootRoot = sqrt(maxRoot)+1;
	// allocate the primes on the device and set them to 0
	cudaMalloc(&d_Primes, sizePrimes);
	cudaMemset(d_Primes, 0, sizePrimes);
	// make sure that there are no errors...
	cudaPeekAtLastError();
	// setup the execution configuration
	dim3 dimBlock1(maxRootRoot, 1, 1);
	dim3 dimBlock2(maxRoot, 1, 1);
	dim3 dimGrid(1);
	//////// debug
	#ifdef DEBUG
	printf("dimBlock1(%d, %d, %d)\n", dimBlock1.x, dimBlock1.y, dimBlock1.z);
	printf("dimBlock2(%d, %d, %d)\n", dimBlock2.x, dimBlock2.y, dimBlock2.z);
	printf("dimGrid(%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
	#endif
	
	// call the kernel
	sieveOfEratosthenes<<<dimGrid, dimBlock1>>>(d_Primes, maxRoot);
	segmentSieve<<<dimGrid, dimBlock2>>>(d_Primes, max);
	
	// check for kernel errors
	cudaPeekAtLastError();
	cudaDeviceSynchronize();
	// copy the results back
	cudaMemcpy(primes, d_Primes, sizePrimes, cudaMemcpyDeviceToHost);
	// no memory leaks
	cudaFree(d_Primes);
}
/**********************************************************************************/
int main(){
	uint64_t maxPrime = 1<<19;//1<<20; // find all primes from 0 to N-1
	char* primes = (char*) malloc(maxPrime);
	memset(primes, 0, maxPrime); // initialize all elements to 0
	genPrimesOnDevice(primes, maxPrime);
	// display the results
	int i;
	int count = 0;
	for (i = 0; i < maxPrime; i++){
		if (primes[i] == 0){
			// printf("%i ", i);
			count++;
		}
	}
	printf("\n");
	printf("max= %i \n", maxPrime);
	printf("number of primes= %i \n", count);
	free(primes);
	return 0;
}

