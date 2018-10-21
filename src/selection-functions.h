#ifndef __SELECTION_FUNCTIONS_H__
#define __SELECTION_FUNCTIONS_H__

#include "stereo-matching.h"

// Winner Take-All
void WTA(StereoMatcher *self);

// CUDA implementation of WTA
void CU_WTA(StereoMatcher *self);

// Total Variation Regularisation
// http://openaccess.thecvf.com/content_iccv_workshops_2013/W21/papers/Kuschk_Fast_and_Accurate_2013_ICCV_paper.pdf
void TVR(StereoMatcher *self);

// CUDA implementation of TVR
void CU_TVR(StereoMatcher *self);


#endif
