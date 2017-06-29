/*
 * BoundingVolumeHierarchy.h
 *
 *  Created on: Apr 1, 2017
 *      Author: julian
 */

#ifndef BOUNDINGVOLUMEHIERARCHY_H_
#define BOUNDINGVOLUMEHIERARCHY_H_

#include "AxisAlignedBoundingBox.h"
#include <list>

using namespace std;

namespace mybrand {

struct Node {
	list<Node> children;
};

struct InternalNode : public Node {

};

struct LeafNode : public Node {

};

class BoundingVolumeHierarchy {
public:
	BoundingVolumeHierarchy();
	virtual ~BoundingVolumeHierarchy();
};

#define BVH BoundingVolumeHierarchy

} /* namespace mybrand */

#endif /* BOUNDINGVOLUMEHIERARCHY_H_ */
