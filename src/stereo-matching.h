#ifndef __STEREO_MATCHING_H__
#define __STEREO_MATCHING_H__

#include "imageio.h"

typedef struct _StereoMatcher
{
  Image *left;
  Image *right;
  Image *disparityImg;
  int disparityRange;
  int feItems;
  float *matchingCostsRead;
  float *matchingCostsWrite;
  float *matchingCostsBuff1;
  float *matchingCostsBuff2;
  void (*smoothFunc)(Image *);
  void (*selectionFunc)(struct _StereoMatcher *);
  void (*aggregationFunc)(struct _StereoMatcher *);
  void (*matchFunc)(struct _StereoMatcher *);
  void (*refineFunc)(struct _StereoMatcher *);

  // ======================
  // <CUDA memory>
  // ======================
  int *d_fe;
  float *d_L;
  float *d_p;
  float *d_q;
  float *d_u;
  float *d_a;
  float *d_uHat;
  float *d_g;
  float *d_leftNorm;
  float *d_grad;
  float *d_gradBard;
  float *d_v;
  float *d_vHat;
  float *d_gradVHat;
  float *d_gradUHat;
  float *d_Gxp;
  // =======================
  // </CUDA memory>
  // =======================
} StereoMatcher;

StereoMatcher *allocMatcher(
  Image *left,
  Image *right,
  int disparityRange,
  char *featureExtractor,
  void (*smoothFunc)(Image *),
  void (*selectionFunc)(StereoMatcher *),
  void (*aggregationFunc)(StereoMatcher *),
  void (*matchFunc)(StereoMatcher *),
  void (*refineFunc)(StereoMatcher *)
);

void normaliseCosts(StereoMatcher *sm);

Image *genDisparityImage(StereoMatcher *sm);

#define FEATURE_EXTRACTOR_ITEMS 64

#define cudaCalloc(A, B) \
  do { \
    cudaError_t __cudaCalloc_err = cudaMalloc(A, B); \
    if (__cudaCalloc_err == cudaSuccess) cudaMemset(*A, 0, B); \
  } while (0)

#define COST_AT_INDEX(i, SM) (SM->matchingCosts[i])
#define COST_AT_INDEX_CUDA(i, matchingCosts) (matchingCosts[i])

// Depth represented by y, height by x, width by disparity. Better peformance
// since reading/writing will be same order as it appears in memory
#define COST_AT(x, y, d, SM) \
  COST_AT_INDEX((d)+SM->disparityRange*((x)+SM->left->width*(y)), SM)
#define COST_AT_CUDA(x, y, d, dimX, dimZ, matchingCosts) \
  COST_AT_INDEX_CUDA(d + dimZ * (x + dimX * y), matchingCosts)

#define SAFE_COST_AT_INDEX(i, SM, undefVal) ( \
  (i) <= SM->left->width * SM->left->height * SM->disparityRange && \
  (i) >= 0 ? COST_AT_INDEX(i, SM) : (undefVal))
#define SAFE_COST_AT_INDEX_CUDA(i, dimX, dimY, dimZ, matchingCosts, undefVal) ( \
  (i) <= (dimX) * (dimY) * (dimZ) && \
  (i) >= 0 ? COST_AT_INDEX_CUDA(i, matchingCosts) : (undefVal))


#define SAFE_COST_AT(x, y, d, SM, undefVal) \
  SAFE_COST_AT_INDEX((d)+SM->disparityRange*((x)+SM->left->width*(y)), SM, undefVal)
#define SAFE_COST_AT_CUDA(x, y, d, dimX, dimY, dimZ, matchingCosts, undefVal) \
  SAFE_COST_AT_INDEX_CUDA(d + (dimZ) * (x + (dimX) * y), (dimX), (dimY), (dimZ), matchingCosts, undefVal)

#endif
