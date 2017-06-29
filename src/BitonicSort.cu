/*
 * BitonicSort.cu
 *
 *  Created on: Apr 9, 2017
 *      Author: julian
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <BitonicSort.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

/* Every thread gets exactly one value in the unsorted array. */
#define BLOCKS 32768 // 2^15, this is a max
#define NUM_VALS THREADS*BLOCKS

/* Every thread gets exactly one value in the unsorted array. */
#define MAX_THREADS_PER_BLOCK 	1024
#define MAX_BLOCKS 				65535
#define MAX_VALUES 				MAX_THREADS_PER_BLOCK * MAX_BLOCKS

void kernel(int* a, int length, int p, int q) {
	int d = 1 << (p - q);

	for (int i = 0; i < length; i++) {
		bool up = ((i >> p) & 2) == 0;

		if ((i & d) == 0 && (a[i] > a[i | d]) == up) {
			int t = a[i];
			a[i] = a[i | d];
			a[i | d] = t;
		}
	}
}

void cpuBitonicSort(int logn, int* array, int length) {
	// assert length == 1 << logn;

	for (int i = 0; i < logn; i++) {
		for (int j = 0; j <= i; j++) {
			kernel(array, length, i, j);
		}
	}
}

//void printElapsed(clock_t start, clock_t stop) {
//	double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
//	printf("Elapsed time: %.3fs\n", elapsed);
//}

float randomFloat() {
	return (float) rand() / (float) RAND_MAX;
}

void arrayPrint(float *arr, int length) {
	int i;
	for (i = 0; i < length; ++i) {
		printf("%1.3f ", arr[i]);
	}
	printf("\n");
}

void arrayFill(float *arr, int length) {
	srand(time(NULL));
	int i;
	for (i = 0; i < length; ++i) {
		arr[i] = randomFloat();
	}
}

__global__ void bitonicSortKernel(int* dArray, int j, int k) {
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i ^ j;

	/* The threads with the lowest ids sort the array. */
	if ((ixj) > i) {
		if ((i & k) == 0) {
			/* Sort ascending */
			if (dArray[i] > dArray[ixj]) {
				/* exchange(i,ixj); */
				float temp = dArray[i];
				dArray[i] = dArray[ixj];
				dArray[ixj] = temp;
			}
		}
		if ((i & k) != 0) {
			/* Sort descending */
			if (dArray[i] < dArray[ixj]) {
				/* exchange(i,ixj); */
				float temp = dArray[i];
				dArray[i] = dArray[ixj];
				dArray[ixj] = temp;
			}
		}
	}
}

/**
 * Inplace bitonic sort using CUDA.
 */
void gpuBitonicSort(int* hArray, int length) {
	int* dArray;

	// TODO Make sure the length is a power of 2.

	int blocks = length / MAX_THREADS_PER_BLOCK;
	int logn = ceil(log(blocks));
	int threads = pow(2, logn);
	cudaMalloc(&dArray, length * sizeof(int));
	cudaMemcpy(dArray, hArray, length * sizeof(int), cudaMemcpyHostToDevice);

	int j, k;
	/* Major step */
	for (k = 2; k <= length; k <<= 1) {
		/* Minor step */
		for (j = k >> 1; j > 0; j = j >> 1) {
			bitonicSortKernel<<<blocks, threads>>>(dArray, j, k);
		}
	}
	cudaDeviceSynchronize();
	cudaMemcpy(hArray, dArray, length * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(dArray);
}

