/*
 * Time.h
 *
 *  Created on: May 28, 2015
 *      Author: julian
 */

#ifndef TIME_H_
#define TIME_H_

#include <reactor/Common.h>
#include <sys/time.h>
#include <time.h>

void getCurrentTime(timespec* ts);

/**
 * Compares two timespec values.  The function returns negative if a > b, positive if b > a, and 0 if equal
 *
 * @param a A timespec to compare
 * @param b A timespec to compare
 */
int8 compareTime(timespec& a, timespec& b);
void subtractTime(timespec& a, timespec& b, timespec& out);

void printElapsed(clock_t start, clock_t stop, const char* message);


#endif /* TIME_H_ */
