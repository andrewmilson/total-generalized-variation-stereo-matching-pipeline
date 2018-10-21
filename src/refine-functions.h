#ifndef __REFINEMENT_FUNCTIONS_H__
#define __REFINEMENT_FUNCTIONS_H__

#include "stereo-matching.h"
#include <stdlib.h>

// Median Filtering (https://en.wikipedia.org/wiki/Median_filter)
void MedianFiltering(StereoMatcher *self);

#endif
