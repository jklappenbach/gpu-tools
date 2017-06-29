/*
 * RadixSort.cpp
 *
 *  Created on: Apr 24, 2014
 *      Author: julian
 */

#include "RadixSort.h"

// Original
void cpuRadixSort(uint32* array, int length, int base) {
	int i;
	uint32 maxValue = array[0];
	uint32 exp = 1;
	uint32* cache = new uint32[length];
	uint32* src;
	uint32* dest;
	bool cacheForward = true;

	// Get the greatest value in the array pArray and assign it to maxValue
	for (i = 1; i < length; i++) {
		if (array[i] > maxValue)
			maxValue = array[i];
	}

	int pow = log(maxValue) / log(base);
	uint32* bucket = new uint32[base];
	int arraySize = base * sizeof(uint32);

	// Loop until exp is bigger than the largest number
	while (pow >= 0) {
		src = cacheForward ? array : cache;
		dest = cacheForward ? cache : array;

		memset(bucket, 0, arraySize);

		// Count the number of keys that will go into each bucket
		for (i = 0; i < length; i++)
			bucket[(src[i] / exp) % base]++;

		// Add the count of the previous buckets to acquire the indexes after the end of each bucket location in the array
		for (i = 1; i < base; i++)
			bucket[i] += bucket[i - 1];

		// Starting at the end of the list, get the index corresponding to the pArray[i]'s key, decrement it,
		// and use it to place pArray[i] into array b.
		for (i = length - 1; i >= 0; i--) {
			uint32 index = --bucket[(src[i] / exp) % base];
			dest[index] = src[i];
		}

		// Multiply exp by the base to get the next group of keys
		exp *= base;
		pow--;
		cacheForward = !cacheForward;
	}

	if (!cacheForward)
		memcpy(array, cache, length * sizeof(uint32));

	delete[] cache;
}

__global__ void maxValueKernel(uint32* dArray, uint32 length, uint32* results, int stride) {
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;
	int begin = threadId * stride;
	int end = begin + stride;
	for (int i = begin; i < end && i < length; i++) {
		if (dArray[i] > results[threadId])
			results[threadId] = dArray[i];
	}
}

/**
 * Write data to the destination, using the
 */
__global__ void rankAndPermuteKernel(uint32* src, uint32* dest, int length, unsigned int* buckets, int threadsPerBlock, int base, int stride, int exp) {
	int begin = blockIdx.x * threadsPerBlock * stride;
	int end = begin + (length / gridDim.x);
	int i, index, bucketIndex;

	end = (end > length ? length : end);

	for (i = end - 1; i >= begin; i--) {
		bucketIndex = (src[i] / exp) % base;
		index = --(*(buckets + (blockIdx.x * base) + bucketIndex));
		dest[index] = src[i];
	}
}

/**
 *
 */
__host__ void scanBuckets(uint32* buckets, int base, int blocks) {
	int sum = 0;
	for (int i = 0; i < base; i++) {
		for (int j = 0; j < blocks; j++) {
			*(buckets + (j * base) + i) += sum;
			sum = *(buckets + (j * base) + i);
		}
	}
}

__global__ void histogramKeysKernel(uint32* src, int length, int stride, uint32* buckets, int base, int exp) {
	int threadId = threadIdx.x + blockDim.x * blockIdx.x;
	int begin = threadId * stride;
	int end = (begin + stride) < length ? begin + stride : length;

	for (int i = begin; i < end; i++) {
		atomicAdd((unsigned int*) buckets + (blockIdx.x * base) + ((src[i] / exp) % base), 1);
	}
}

__host__ void gpuRadixSort(uint32* hArray, int length, int base, int stride) {
	bool cacheForward = true;
	int totalThreads = stride < length ? ceil(length / stride) : 1;
	int threadsPerBlock = totalThreads > MAX_THREADS_PER_BLOCK ? MAX_THREADS_PER_BLOCK : totalThreads;
	int blocks = ceil((double) length / (double) (threadsPerBlock * stride));

	// Copy the array into the GPU memory
	uint32* dArray;
	checkCudaErrors(cudaMalloc(&dArray, length * sizeof(uint32) * 2));
	checkCudaErrors(cudaMemcpy(dArray, hArray, length * sizeof(uint32), cudaMemcpyHostToDevice));

	// Max value MR...
	uint32* dMaxResults;
	uint32* hMaxResults;
	checkCudaErrors(cudaMalloc(&dMaxResults, totalThreads * sizeof(uint32)));
	checkCudaErrors(cudaMemset(dMaxResults, 0, totalThreads * sizeof(uint32)));
	checkCudaErrors(cudaHostAlloc(&hMaxResults, totalThreads * sizeof(uint32), cudaHostAllocDefault));
	maxValueKernel<<<blocks, threadsPerBlock>>>(dArray, length, dMaxResults, stride);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(hMaxResults, dMaxResults, totalThreads * sizeof(uint32), cudaMemcpyDeviceToHost));
	int maxValue = hMaxResults[0];
	for (int i = 0; i < totalThreads; i++) {
		if (maxValue < hMaxResults[i])
			maxValue = hMaxResults[i];
	}

	unsigned int exp = 1;
	unsigned int* dBuckets;
	unsigned int* hBuckets;
	checkCudaErrors(cudaHostAlloc(&hBuckets, blocks * base * sizeof(uint32), cudaHostAllocDefault));
	checkCudaErrors(cudaMalloc(&dBuckets, blocks * base * sizeof(uint32)));
	checkCudaErrors(cudaMalloc((void**)&dBuckets, blocks * base * sizeof(uint32)));
	uint32* src;
	uint32* dest;

	int pow = log(maxValue) / log(base);

	// Loop until exp is bigger than the largest number
	while (pow >= 0) {
		src = cacheForward ? dArray : dArray + length;
		dest = cacheForward ? dArray + length : dArray;
		checkCudaErrors(cudaMemset(dBuckets, 0, blocks * base * sizeof(uint32)));
		histogramKeysKernel<<<blocks, threadsPerBlock>>>(src, length, stride, dBuckets, base, exp);
		checkCudaErrors(cudaDeviceSynchronize());
		checkCudaErrors(cudaMemcpy(hBuckets, dBuckets, blocks * base * sizeof(uint32), cudaMemcpyDeviceToHost));
		scanBuckets(hBuckets, base, blocks);
		checkCudaErrors(cudaMemcpy(dBuckets, hBuckets, blocks * base * sizeof(uint32), cudaMemcpyHostToDevice));
		rankAndPermuteKernel<<<blocks, 1>>>(src, dest, length, dBuckets, threadsPerBlock, base, stride, exp);
		checkCudaErrors(cudaDeviceSynchronize());
		cacheForward = !cacheForward;
		exp *= base;
		pow--;
	}

	cudaMemcpy(hArray, dArray + (cacheForward ? 0 : length), length * sizeof(uint32), cudaMemcpyDeviceToHost);
	checkCudaErrors(cudaFreeHost(hBuckets));
	checkCudaErrors(cudaFree(dBuckets));
	checkCudaErrors(cudaFreeHost(hMaxResults));
	checkCudaErrors(cudaFree(dMaxResults));
	checkCudaErrors(cudaFree(dArray));
}


