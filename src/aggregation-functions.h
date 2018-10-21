#ifndef __AGGREGATION_FUNCTIONS_H__
#define __AGGREGATION_FUNCTIONS_H__

#include "stereo-matching.h"

// Locally Adaptive Support-Weight Approach for Visual Correspondence Search
// https://github.com/CheerM/opencv_stereo_matching/blob/master/%5B2005%20CVPR%5D%20Locally%20Adaptive%20Support-Weight%20Approach%20for%20Visual%20Correspondence%20Search.pdf
void localAdaptiveSupportWeight(StereoMatcher *self);
void CU_localAdaptiveSupportWeight(StereoMatcher *self);

#endif
