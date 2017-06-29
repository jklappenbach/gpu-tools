/*
 * main.cpp
 *
 *  Created on: Apr 4, 2017
 *      Author: julian
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <RadixSort.h>
#include <BitonicSort.h>
#include <QuickSort.h>
#include <reactor/Time.h>
#include <BinaryRadixTree.h>
#include <reactor/Common.h>

void writeBits(FILE* file, uint32 number) {
	for (uint32 mask = ~(~0u >> 1); mask > 0; mask >>= 1) {
		char c = (number & mask ? '1' : '0');
		fprintf(file, "%c", c);
	}
}

void writeBinaryRadixResults(const char* filePath, BinaryRadixNode* nodes, uint32* keys, int length) {
	FILE* file = fopen(filePath, "w+");

	fprintf(file, "Nodes:\n");
	for (int i = 0; i < length - 1; i++) {
		fprintf(file, "Node Index: %d, i: %d, j: %d, left: %d, leftLeaf: %s, right: %d, rightLeaf: %s\n",
				i, (nodes)[i].i, (nodes)[i].j, (nodes)[i].left, (nodes)[i].leftLeaf ? "true" : "false",
						(nodes)[i].right, (nodes)[i].rightLeaf ? "true": "false");
	}
	fprintf(file, "\n\nArray Format:\n");
	fprintf(file, "{\n");
	for (int i = 0; i < length; i++) {
		fprintf(file, "    %u%s\t// ", keys[i], i < length - 1 ? "," : "");
		writeBits(file, keys[i]);
		fprintf(file, "\n");
	}
	fprintf(file, "}\n");
	fclose(file);
}

uint64* createBitArray(uint32 length) {
	int words = ceil((double) length / 64.0);
	uint64* array = new uint64[words];
	memset(array, 0, words * 8);
}

void setBitArray(uint64* array, uint32 index, bool value) {
	int word = index / 64;
	int bitInWord = 1 << (index % 64);
	array[word] |= bitInWord;
}

bool getBitArray(uint64* array, uint32 index) {
	bool result;
	int word = index / 64;
	int bitInWord = 1 << (index % 64);
	return array[word] & bitInWord;
}

bool checkBitArraySet(uint64* array, uint32 length) {
	int words = ceil((double) length / 64.0);
	int i;
	for (i = 0; i < words - 1; i++) {
		if (array[i] != 0xFFFFFFFFFFFFFFFF) {
			return false;
		}
	}
	// Now we're on the last word, which may be partially set.
	// We'll look at it one bit at a time.
	int bits = length % 64;
	for (uint64 j = 0, mask = 1; j < bits; j++, mask <<= 1) {
		if (!(array[i] & mask)) {
			return false;
		}
	}

	return true;
}

/**
 * Recurse through the tree, marking each key as they're discovered as leaves in the tree
 */
void recurseNode(BinaryRadixNode* nodes, int nodeId, uint64* bitArray, int depth = 0) {
	//printf("recurseNode depth: %d\n", depth);
	if (depth > 12)
		int x = 0;

	if (nodes[nodeId].leftLeaf) {
		setBitArray(bitArray, nodes[nodeId].left, true);
	} else {
		recurseNode(nodes, nodes[nodeId].left, bitArray, depth + 1);
	}

	if (nodes[nodeId].rightLeaf) {
		setBitArray(bitArray, nodes[nodeId].right, true);
	} else {
		recurseNode(nodes, nodes[nodeId].right, bitArray, depth + 1);
	}
}

bool verifySort(uint32* array, uint32 length) {
	for (uint32 i = 1; i < length; i++) {
		if (array[i - 1] > array[i])
			return false;
	}
	return true;
}
bool verifyTree(BinaryRadixNode* nodes, int length) {
	uint64* bitArray = createBitArray(length);
	recurseNode(nodes, 0, bitArray);
	bool result = checkBitArraySet(bitArray, length);
	delete[] bitArray;
	return result;
}


void sortTests() {
	clock_t start, stop;
	int logn = 23, length = 1 << logn;
	int base = 2048;
	int stride = 32;
	int maxValue = RAND_MAX;

	uint32* hCpuRadixArray = new uint32[length];
	uint32* hGpuRadixArray = new uint32[length];

	srand(time(NULL));
	for (int i = 0; i < length; i++) {
		hCpuRadixArray[i] = hGpuRadixArray[i] =
					((double) rand() / (double) RAND_MAX * maxValue);;
		if (hGpuRadixArray[i] < 0)
			printf("\n\nWe have less than zero.\n\n");
	}

	printf("%d Elements, %d Base, %d Stride\n", length, base, stride);

//	start = clock();
//	cpuBitonicSort(logn, hCpuBitonicArray, length);
//	stop = clock();
//	printElapsed(start, stop, "CPU Bitonic Sort: ");

	start = clock();
	cpuRadixSort(hCpuRadixArray, length, base);
	stop = clock();
	printElapsed(start, stop, "CPU Radix Sort: ");

//	start = clock();
//	cpuQuickSort(hCpuQuickArray, length);
//	stop = clock();
//	printElapsed(start, stop, "CPU Quick Sort: ");

	start = clock();
	gpuRadixSort(hGpuRadixArray, length, base, stride);
	stop = clock();
	printElapsed(start, stop, "GPU Radix Sort: ");

	verifySort(hCpuRadixArray, length);
	verifySort(hGpuRadixArray, length);

	// Now, let's treat our keys as if it was a set of morton codes

	delete[] hGpuRadixArray;
	delete[] hCpuRadixArray;

	printf("\nComplete.");
}

void binaryRadixTreeUnitTest8() {
	int length = 8;
	uint32 keys[] = {
		0b00001000000000000000000000000000, // 134217728
		0b00010000000000000000000000000000, // 268435456
		0b00100000000000000000000000000000, // 536870912
		0b00101000000000000000000000000000, // 671088640
		0b10011000000000000000000000000000, // 2550136832
		0b11000000000000000000000000000000, // 3221225472
		0b11001000000000000000000000000000, // 3355443200
		0b11110000000000000000000000000000  // 4026531840
	};
	BinaryRadixNode* nodes = null;
	clock_t start, stop;
	printf("\nStarting build for 8 element Binary Radix Tree");
	start = clock();
	buildBinaryRadixTree(keys, length, &nodes);
	stop = clock();
	printElapsed(start, stop, "Total time, including data transfer: ");
	if (verifyTree(nodes, length)) {
		printf("\nTree passed verification at %d elements.\n", length);
	} else {
		printf("\nTree failed verification at %d elements.\n", length);
	}
	writeBinaryRadixResults("binary-radix-tree-8.txt", nodes, keys, length);
	delete[] nodes;
}

void binaryRadixTreeUnitTest20() {
	int length = 20;

//	Node Index: 0, i: 0, j: 19, left: 12, leftLeaf: false, right: 13, rightLeaf: false
//	Node Index: 1, i: 1, j: 0, left: 0, leftLeaf: true, right: 1, rightLeaf: true
//	Node Index: 2, i: 2, j: 3, left: 2, leftLeaf: true, right: 3, rightLeaf: true
//	Node Index: 3, i: 3, j: 0, left: 1, leftLeaf: false, right: 2, rightLeaf: false
//	Node Index: 4, i: 4, j: 6, left: 4, leftLeaf: true, right: 5, rightLeaf: false
//	Node Index: 5, i: 5, j: 6, left: 5, leftLeaf: true, right: 6, rightLeaf: true
//	Node Index: 6, i: 6, j: 0, left: 3, leftLeaf: false, right: 4, rightLeaf: false
//	Node Index: 7, i: 7, j: 12, left: 9, leftLeaf: false, right: 10, rightLeaf: false
//	Node Index: 8, i: 8, j: 9, left: 8, leftLeaf: true, right: 9, rightLeaf: true
//	Node Index: 9, i: 9, j: 7, left: 7, leftLeaf: true, right: 8, rightLeaf: false
//	Node Index: 10, i: 10, j: 12, left: 11, leftLeaf: false, right: 12, rightLeaf: true
//	Node Index: 11, i: 11, j: 10, left: 10, leftLeaf: true, right: 11, rightLeaf: true
//	Node Index: 12, i: 12, j: 0, left: 6, leftLeaf: false, right: 7, rightLeaf: false
//	Node Index: 13, i: 13, j: 19, left: 15, leftLeaf: false, right: 16, rightLeaf: false
//	Node Index: 14, i: 14, j: 13, left: 13, leftLeaf: true, right: 14, rightLeaf: true
//	Node Index: 15, i: 15, j: 13, left: 14, leftLeaf: false, right: 15, rightLeaf: true
//	Node Index: 16, i: 16, j: 19, left: 17, leftLeaf: false, right: 18, rightLeaf: false
//	Node Index: 17, i: 17, j: 16, left: 16, leftLeaf: true, right: 17, rightLeaf: true
//	Node Index: 18, i: 18, j: 19, left: 18, leftLeaf: true, right: 19, rightLeaf: true

	uint32 keys[] = {
		    0,		// 00000000000000000000000000000000
		    10,		// 00000000000000000000000000001010
		    20,		// 00000000000000000000000000010100
		    30,		// 00000000000000000000000000011110
		    40,		// 00000000000000000000000000101000
		    50,		// 00000000000000000000000000110010
		    60,		// 00000000000000000000000000111100
		    70,		// 00000000000000000000000001000110
		    80,		// 00000000000000000000000001010000
		    90,		// 00000000000000000000000001011010
		    100,	// 00000000000000000000000001100100
		    110,	// 00000000000000000000000001101110
		    120,	// 00000000000000000000000001111000
		    130,	// 00000000000000000000000010000010
		    140,	// 00000000000000000000000010001100
		    150,	// 00000000000000000000000010010110
		    160,	// 00000000000000000000000010100000
		    170,	// 00000000000000000000000010101010
		    180,	// 00000000000000000000000010110100
		    190		// 00000000000000000000000010111110
	};

	BinaryRadixNode* nodes = null;
	clock_t start, stop;
	printf("\nStarting build for 20 element Binary Radix Tree");
	start = clock();
	buildBinaryRadixTree(keys, length, &nodes);
	stop = clock();
	printElapsed(start, stop, "Total time, including data transfer: ");
	if (verifyTree(nodes, length)) {
		printf("\nTree passed verification at %d elements.\n", length);
	} else {
		printf("\nTree failed verification at %d elements.\n", length);
	}
	writeBinaryRadixResults("binary-radix-tree-20.txt", nodes, keys, length);
	delete[] nodes;
}

void binaryRadixTreeDynamicTest() {
	int base = 2048, stride = 32;
	int length = 300000;
	uint32* keys = new uint32[length];
	srand(time(null));
	for (int i = 0; i < length; i++) {
		keys[i] = i;
	}

	printf("\nStarting build for %d element Binary Radix Tree", length);
	clock_t start, stop;
	start = clock();
	BinaryRadixNode* nodes = null;
	buildBinaryRadixTree(keys, length, &nodes);
	stop = clock();

	printElapsed(start, stop, "Total time, including data transfer: ");

	if (verifyTree(nodes, length)) {
		printf("\nTree passed verification at %d elements.\n", length);
	} else {
		printf("\nTree failed verification at %d elements.\n", length);
	}
	writeBinaryRadixResults("binary-radix-tree-dynamic.txt", nodes, keys, length);
	delete[] nodes;
}

int main(int argc, char** argv) {
//	binaryRadixTreeUnitTest8();
//	binaryRadixTreeUnitTest20();
	binaryRadixTreeDynamicTest();
}



