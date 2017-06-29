/*
 * AxisAlignedBoundingBox.h
 *
 *  Created on: Apr 1, 2017
 *      Author: julian
 */

#ifndef AXISALIGNEDBOUNDINGBOX_H_
#define AXISALIGNEDBOUNDINGBOX_H_

struct AxisAlignedBoundingBox {
	float minX, maxX, minY, maxY, minZ, maxZ;
};

__host__ AxisAlignedBoundingBox aabbUnion(AxisAlignedBoundingBox arg1, AxisAlignedBoundingBox arg2);

#endif /* AXISALIGNEDBOUNDINGBOX_H_ */
