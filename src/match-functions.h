#ifndef __MATCH_FUNCTIONS_H__
#define __MATCH_FUNCTIONS_H__

#include "stereo-matching.h"

void CU_census(StereoMatcher *self);

void CU_binaryFilter(StereoMatcher *self);

// Census transform
void census(StereoMatcher *self);

// Sum of Absolute Differences in CUDA
void CU_SAD(StereoMatcher *self);

// Sum of Absolute Differences
void SAD(StereoMatcher *self);

// Sum of Squared Differences
void SSD(StereoMatcher *self);

#endif
