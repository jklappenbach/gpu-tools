/*
 * BinaryRadixTree.h
 *
 *  Created on: May 6, 2017
 *      Author: julian
 */

#ifndef BINARYRADIXTREE_H_
#define BINARYRADIXTREE_H_

struct BinaryRadixNode {
	uint32 i;
	uint32 j;
	uint32 parent;
	uint32 left;
	bool leftLeaf;
	uint32 right;
	bool rightLeaf;
};

__host__ void buildBinaryRadixTree(uint32* hKeys, int32 length, BinaryRadixNode** nodes);

#endif /* BINARYRADIXTREE_H_ */
