/*
 * AxisAlignedBoundingBox.cpp
 *
 *  Created on: Apr 1, 2017
 *      Author: julian
 */

#include <AxisAlignedBoundingBox.h>

__host__ AxisAlignedBoundingBox aabbUnion(AxisAlignedBoundingBox arg1, AxisAlignedBoundingBox arg2) {
	AxisAlignedBoundingBox result;
	result.minX = min(arg1.minX, arg2.minX);
	result.maxX = max(arg1.maxX, arg2.maxX);
	result.minY = min(arg1.minY, arg2.minY);
	result.maxY = max(arg1.maxY, arg2.maxY);
	result.minZ = min(arg1.minZ, arg2.minZ);
	result.maxZ = max(arg1.maxZ, arg2.maxZ);
	return result;
}
