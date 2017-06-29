/*
 * BoundingVolumeHiearchy.h
 *
 *  Created on: Apr 1, 2017
 *      Author: julian
 */

#ifndef BOUNDINGVOLUMEHIEARCHY_CU_
#define BOUNDINGVOLUMEHIEARCHY_CU_

#include "BoundingVolumeHierarchy.h"

namespace mybrand {

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

//Node* generateHierarchy(unsigned int* sortedMortonCodes, int* sortedObjectIDs,
//		int first, int last) {
//	// Single object => create a leaf node.
//
//	if (first == last)
//		return new LeafNode(&sortedObjectIDs[first]);
//
//	// Determine where to split the range.
//	int split = findSplit(sortedMortonCodes, first, last);
//
//	// Process the resulting sub-ranges recursively.
//	Node* childA = generateHierarchy(sortedMortonCodes, sortedObjectIDs, first,
//			split);
//	Node* childB = generateHierarchy(sortedMortonCodes, sortedObjectIDs,
//			split + 1, last);
//
//	return new InternalNode(childA, childB);
//}
//
////Node* generateHierarchy(unsigned int* sortedMortonCodes, int* sortedObjectIDs,
////		int first, int last) {
////	// Single object => create a leaf node.
////	if (first == last)
////		return new LeafNode(&sortedObjectIDs[first]);
////
////	// Determine where to split the range.
////	int split = findSplit(sortedMortonCodes, first, last);
////
////	// Process the resulting sub-ranges recursively.
////	Node* childA = generateHierarchy(sortedMortonCodes, sortedObjectIDs, first, split);
////	Node* childB = generateHierarchy(sortedMortonCodes, sortedObjectIDs, split + 1, last);
////	return new InternalNode(childA, childB);
////}
//
//Node* generateHierarchy(unsigned int* sortedMortonCodes, int* sortedObjectIDs, int numObjects) {
//	LeafNode* leafNodes = new LeafNode[numObjects];
//	InternalNode* internalNodes = new InternalNode[numObjects - 1];
//
//	// Construct leaf nodes.
//	// Note: This step can be avoided by storing
//	// the tree in a slightly different way.
//
//	for (int idx = 0; idx < numObjects; idx++) // in parallel
//		leafNodes[idx].objectID = sortedObjectIDs[idx];
//
//	// Construct internal nodes.
//
//	for (int idx = 0; idx < numObjects - 1; idx++) { // in parallel
//		// Find out which range of objects the node corresponds to.
//		// (This is where the magic happens!)
//		int2 range = determineRange(sortedMortonCodes, numObjects, idx);
//		int first = range.x;
//		int last = range.y;
//
//		// Determine where to split the range.
//
//		int split = findSplit(sortedMortonCodes, first, last);
//
//		// Select childA.
//
//		Node* childA;
//		if (split == first)
//			childA = &leafNodes[split];
//		else
//			childA = &internalNodes[split];
//
//		// Select childB.
//
//		Node* childB;
//		if (split + 1 == last)
//			childB = &leafNodes[split + 1];
//		else
//			childB = &internalNodes[split + 1];
//
//		// Record parent-child relationships.
//
//		internalNodes[idx].childA = childA;
//		internalNodes[idx].childB = childB;
//		childA->parent = &internalNodes[idx];
//		childB->parent = &internalNodes[idx];
//	}
//
//	// Node 0 is the root.
//
//	return &internalNodes[0];
//}
//
//__device__ void traverseRecursive(CollisionList& list,
//		const BoundingVolumeHierarchy& bvh, const AABB& queryAABB,
//		int queryObjectIdx, NodePtr node) {
//	// Bounding box overlaps the query => process node.
//	if (checkOverlap(bvh.getAABB(node), queryAABB)) {
//		// Leaf node => report collision.
//		if (bvh.isLeaf(node))
//			list.add(queryObjectIdx, bvh.getObjectIdx(node));
//
//		// Internal node => recurse to children.
//		else {
//			NodePtr childL = bvh.getLeftChild(node);
//			NodePtr childR = bvh.getRightChild(node);
//			traverseRecursive(bvh, list, queryAABB, queryObjectIdx, childL);
//			traverseRecursive(bvh, list, queryAABB, queryObjectIdx, childR);
//		}
//	}
//}
//
//__global__ void findPotentialCollisions(CollisionList list, BVH bvh,
//		AABB* objectAABBs, int numObjects) {
//	int idx = threadIdx.x + blockDim.x * blockIdx.x;
//	if (idx < numObjects)
//		traverseRecursive(bvh, list, objectAABBs[idx], idx, bvh.getRoot());
//
}

#endif /* BOUNDINGVOLUMEHIEARCHY_CU_ */
