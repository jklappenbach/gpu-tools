/*
 * RadixSort.h
 *
 *  Created on: Apr 24, 2014
 *      Author: julian
 */

#ifndef RADIXSORT_H_
#define RADIXSORT_H_

#include <cmath>
#include <reactor/Common.h>
#include <helper_cuda.h>
#include <stdio.h>
#include <limits.h>
#include <reactor/Time.h>


#define MAX_THREADS_PER_BLOCK	1024

void cpuRadixSort(uint32* array, int length, int base);
void gpuRadixSort(uint32* hArray, int length, int base, int stride);
__global__ void maxValueKernel(uint32* dArray, uint32 length, uint32* results, int stride);
__global__ void rankAndPermuteKernel(uint32* src, uint32* dest, int length, unsigned int* buckets, int threadsPerBlock, int base, int stride, int exp);
__host__ void scanBuckets(uint32* buckets, int base, int blocks);
__global__ void histogramKeysKernel(uint32* src, int length, int stride, uint32* buckets, int base, int exp);


#endif /* RADIXSORT_H_ */
