/*
 * ParallelPrefixSum.h
 *
 *  Created on: May 2, 2017
 *      Author: julian
 */

#ifndef PARALLELPREFIXSUM_H_
#define PARALLELPREFIXSUM_H_

#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
	#define CONFLICT_FREE_OFFSET(n) ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
	#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__global__ void prescan(float *g_odata, float *g_idata, int n);

#endif /* PARALLELPREFIXSUM_H_ */
