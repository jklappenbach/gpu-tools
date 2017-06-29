/*
 * Time.cpp
 *
 *  Created on: May 28, 2015
 *      Author: julian
 */
#include <reactor/Time.h>
#include <stdio.h>

#ifdef __DARWIN_UNIX03
	#include <mach/mach_time.h>
	#define ORWL_NANO 		(+1.0E-9)
	#define ORWL_GIGA 		uint64(1000000000)
#else
	#include <sys/time.h>
	#include <time.h>
#endif

void getCurrentTime(timespec* ts) {
#ifdef __DARWIN_UNIX03
	static double orwl_timebase = 0.0;
	static uint64_t orwl_timestart = 0;

	  if (!orwl_timestart) {
		mach_timebase_info_data_t tb = { 0 };
		mach_timebase_info(&tb);
		orwl_timebase = tb.numer;
		orwl_timebase /= tb.denom;
		orwl_timestart = mach_absolute_time();
	  }
	  double diff = (mach_absolute_time() - orwl_timestart) * orwl_timebase;
	  ts->tv_sec = diff * ORWL_NANO;
	  ts->tv_nsec = diff - (ts->tv_sec * ORWL_GIGA);
#else
	  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, ts);
#endif
}

int8 compareTime(timespec& a, timespec& b) {
	if (a.tv_sec > b.tv_sec)
		return -1;
	else if (a.tv_sec < b.tv_sec)
		return 1;
	else {
		if (a.tv_nsec > b.tv_nsec)
			return -1;
		else if (a.tv_nsec < b.tv_nsec)
			return 1;
		else
			return 0;
	}
}

void subtractTime(timespec& a, timespec& b, timespec& out) {
	out.tv_sec = a.tv_sec - b.tv_sec;
	if (b.tv_nsec > a.tv_nsec && a.tv_sec > 0) {
		out.tv_sec -= 1;
		out.tv_nsec = (a.tv_nsec + 1000000000l) - b.tv_nsec;
	} else {
		out.tv_nsec = a.tv_nsec - b.tv_nsec;
	}
}

void printElapsed(clock_t start, clock_t stop, const char* message) {
	double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
	printf("%s: %.3fs\n", message, elapsed);
}





