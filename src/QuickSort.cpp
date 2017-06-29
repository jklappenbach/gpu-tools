/*
 * QuickSort.cpp
 *
 *  Created on: Apr 24, 2014
 *      Author: julian
 */

#include "QuickSort.h"

/**
 *
 *	left is the index of the leftmost element of the subarray
 *	right is the index of the rightmost element of the subarray (inclusive)
 *	number of elements in subarray = right-left+1
 *
 *	function partition(array, left, right, pivotIndex)
 */
int partition(int* pArray, int left, int right) {
	int pivotValue = pArray[right];

	while (left < right) {
		while (pArray[left] < pivotValue)
			left++;
		while (pArray[right] > pivotValue)
			right--;

		if (pArray[left] == pArray[right]) {
			left++;
		} else if (left < right) {
			pArray[left] ^= pArray[right];
			pArray[right] ^= pArray[left];
			pArray[left] ^= pArray[right];

		}
	}
	return right;
}

/**
 * function quicksort(array, left, right)
 *    // If the list has 2 or more items
 *    if left < right
 *        // See "#Choice of pivot" section below for possible choices
 *        choose any pivotIndex such that left ≤ pivotIndex ≤ right
 *        // Get lists of bigger and smaller items and final position of pivot
 *        pivotNewIndex := partition(array, left, right, pivotIndex)
 *        // Recursively sort elements smaller than the pivot (assume pivotNewIndex - 1 does not underflow)
 *        quicksort(array, left, pivotNewIndex - 1)
 *        // Recursively sort elements at least as big as the pivot (assume pivotNewIndex + 1 does not overflow)
 *        quicksort(array, pivotNewIndex + 1, right)
 */
void quickSort(int* pArray, int left, int right) {
	if (left < right) {
		int pivotIndex = partition(pArray, left, right);
		quickSort(pArray, left, pivotIndex - 1);
		quickSort(pArray, pivotIndex + 1, right);
	}
}

void cpuQuickSort(int* pArray, int length) {
	quickSort(pArray, 0, length - 1);
}





