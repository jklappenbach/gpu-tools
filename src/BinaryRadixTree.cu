/*
 * bsp.cpp
 *
 *  Created on: Apr 22, 2017
 *      Author: julian
 */

#include <algorithm>
#include <reactor/Common.h>
#include <helper_cuda.h>
#include <RadixSort.h>
#include <BinaryRadixTree.h>
#include <reactor/Common.h>



using namespace std;

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
unsigned int expandBits(unsigned int v) {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
unsigned int morton3D(float x, float y, float z) {
	x = min(max(x * 1024.0f, 0.0f), 1023.0f);
	y = min(max(y * 1024.0f, 0.0f), 1023.0f);
	z = min(max(z * 1024.0f, 0.0f), 1023.0f);
	unsigned int xx = expandBits((unsigned int) x);
	unsigned int yy = expandBits((unsigned int) y);
	unsigned int zz = expandBits((unsigned int) z);
	return xx * 4 + yy * 2 + zz;
}

__host__ int commonPadding(int i, int j, uint32* hKeys) {
	int result = 0;

	for (uint32 mask = ~(~0u >> 1); mask > 0; mask >>= 1, result++) {
		if ((hKeys[i] & mask) != (hKeys[j] & mask)) {
			return result;
		}
	}

	return result;
}

__device__ int commonBitPrefix(int i, int j, uint32* dKeys, int padding) {
	int result = 0;

	for (uint32 mask = ~(~0u >> 1) >> padding; mask > 0; mask >>= 1, result++) {
		if ((dKeys[i] & mask) != (dKeys[j] & mask)) {
			return result;
		}
	}

	return result;
}

//
//
//

__global__ void buildBinaryRadixKernel(uint32* dKeys, BinaryRadixNode* nodes, int32 length,
		int32 padding) {
	int32 i = threadIdx.x + blockDim.x * blockIdx.x;

	if (i < length - 1) {
		int32 j = length - 1;
		int32 direction = 1;
		uint32 nodePrefixLength = 0;
		int adjacent = 0;
		int minPrefixLength = 0;
		int prefixLength = 0;
		uint32 partition;
		int32 delta;
		int left, right;

		// If we're at the root node, we know (i, j) <-- (0, length - 1)
		if (i > 0) {
			// Find the span of the node
			left = commonBitPrefix(i, i - 1, dKeys, padding);
			right = commonBitPrefix(i, i + 1, dKeys, padding);
			direction = left < right ? 1 : -1;
			minPrefixLength = direction > 0 ? left : right;

			// Find the maximum extent using an ascending binary search for j
			// We'll assume we've probably gone a bit too far
			delta = direction;
			j = i;

			do {
				delta *= 2;
				j += delta;
				j = (j < 0 || j > length - 1) ? (j < 0 ? 0 : length - 1) : j;
				prefixLength = commonBitPrefix(i, j, dKeys, padding);
			} while (j > 0 && j < (length - 1) && prefixLength > minPrefixLength);

			adjacent = j + direction > length - 1 || j + direction < 0 ? 0 :
					commonBitPrefix(i, j + direction, dKeys, padding);

			// Now, dial it back using a descending binary search
			while (prefixLength <= minPrefixLength || (prefixLength > minPrefixLength &&
					adjacent > minPrefixLength)) {
				delta = abs(delta) / (prefixLength > minPrefixLength ? (abs(delta) > 1 ? 2 : 1) :
						(abs(delta) > 1 ? -2 : -1)) * direction;
				j += delta;
				j = (j < 0 || j > length - 1) ? (j < 0 ? 0 : length - 1) : j;
				prefixLength = commonBitPrefix(i, j, dKeys, padding);
				adjacent = j == length - 1 || j == 0 ? 0 :
						commonBitPrefix(i, j + direction, dKeys, padding);
			}
			nodePrefixLength = prefixLength;
		} else {
			nodes[0].parent = 0;
			nodePrefixLength = commonBitPrefix(0, length - 1, dKeys, padding);
		}

		// The node range should now be defined by (i, j).  If the length
		// of the range is greater than 1, find the partition within.
		int k = min(i, j);
		int l = max(i, j);
		if (abs(k - l) > 1) {
			partition = l;
			delta = l - k;
			adjacent = (l == length - 1) ? 0 : commonBitPrefix(k, partition + 1, dKeys, padding);
			while (prefixLength == nodePrefixLength || (prefixLength > nodePrefixLength &&
					adjacent > nodePrefixLength)) {
				delta = abs(delta) / (prefixLength == nodePrefixLength ?
						(abs(delta) > 1 ? -2 : -1) : (abs(delta) > 1 ? 2 : 1));
				partition += delta;
				prefixLength = commonBitPrefix(k, partition, dKeys, padding);
				adjacent = commonBitPrefix(k, partition + 1, dKeys, padding);
			}
		} else {
			partition = k;
		}

		nodes[i].i = i;
		nodes[i].j = j;
		nodes[i].left = partition;
		nodes[i].leftLeaf = (partition == k);
		if (!nodes[i].leftLeaf)
			nodes[nodes[i].left].parent = i;
		nodes[i].right = partition + 1;
		nodes[i].rightLeaf = (nodes[i].right == l);
		if (!nodes[i].rightLeaf) {
			nodes[nodes[i].right].parent = i;
		}
	}
}

__host__ void buildBinaryRadixTree(uint32* hKeys, int32 length, BinaryRadixNode** nodes) {
	int threadsPerBlock = 1024;
	// TODO Input validation here
	BinaryRadixNode* dNodes;
	checkCudaErrors(cudaMalloc(&dNodes, sizeof(BinaryRadixNode) * (length - 1)));
	checkCudaErrors(cudaMemset(dNodes, 0, sizeof(BinaryRadixNode) * (length - 1)));
	BinaryRadixNode* temp = new BinaryRadixNode[length - 1];
	cpuRadixSort(hKeys, length, 2048);

	// Figure out how much padding we have over our array
	int padding = commonPadding(0, length - 1, hKeys);

	uint32* dKeys;
	checkCudaErrors(cudaMalloc(&dKeys, sizeof(uint32) * length));
	checkCudaErrors(cudaMemcpy(dKeys, hKeys, sizeof(uint32) * length, cudaMemcpyHostToDevice));
	int blocks = (int) ceil((double) (length - 1) / (double) threadsPerBlock);
	// TODO NEED TO FIX THIS, WE'RE LAUNCHING TOO MANY THREADS!!!!  AND WE NEED TO BETTER OPTIMIZE FOR SMALLER RUNS!
	// NEED A BALANCE BETWEEN BLOCKS (SMs) RUNNING INDEPENDENTLY
	// AND THREADING GOODNESS!
	int threads = (length - 1) >= threadsPerBlock ? threadsPerBlock : (length - 1);
	clock_t start, stop;
	start = clock();
	buildBinaryRadixKernel<<<blocks, threads>>>(dKeys, dNodes, length, padding);
	checkCudaErrors(cudaDeviceSynchronize());
	stop = clock();
	printElapsed(start, stop, "Kernel Time");

	checkCudaErrors(cudaMemcpy(temp, dNodes, sizeof(BinaryRadixNode) * (length - 1), cudaMemcpyDeviceToHost));
	*nodes = temp;
}
