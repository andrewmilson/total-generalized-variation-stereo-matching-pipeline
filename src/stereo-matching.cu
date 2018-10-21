#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "stereo-matching.h"
#include "logger.h"

int countChar(char* string, char ch) {
  int count = 0;
  int len = strlen(string);

  for (int i = 0; i < len; i++)
    if (string[i] == ch) count++;

  return count;
}

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
) {
    StereoMatcher *sm = (StereoMatcher *) malloc(sizeof(StereoMatcher));

    // Parse the binary filter
    int *fe = (int*) malloc(sizeof(int) * FEATURE_EXTRACTOR_ITEMS * 2);
    char *end = featureExtractor;
    int count = 0;

    while(*end) {
      int n = strtol(featureExtractor, &end, 10);
      fe[count / 2 + (count % 2) * FEATURE_EXTRACTOR_ITEMS] = n;
      count++;

      while (*end == ',') end++;
      featureExtractor = end;
    }

    // Allocate CUDA memory for feature extractor
    cudaCalloc((void **) &sm->d_fe, sizeof(int) * FEATURE_EXTRACTOR_ITEMS * 2);
    cudaMemcpy(sm->d_fe, fe, sizeof(int) * FEATURE_EXTRACTOR_ITEMS * 2,
      cudaMemcpyHostToDevice);

    sm->left = left;
    sm->right = right;

    int imgLen = sm->left->width * sm->left->height;
    int disparityLen = imgLen * disparityRange;

    cudaMalloc((void **) &sm->matchingCostsBuff1, sizeof(float) * disparityLen);
    cudaMalloc((void **) &sm->matchingCostsBuff2, sizeof(float) * disparityLen);
    sm->matchingCostsRead = sm->matchingCostsBuff1;
    sm->matchingCostsWrite = sm->matchingCostsBuff2;

    sm->disparityImg = allocImg(sm->left->width, sm->left->height);
    sm->disparityRange = disparityRange;
    sm->smoothFunc = smoothFunc;
    sm->selectionFunc = selectionFunc;
    sm->aggregationFunc = aggregationFunc;
    sm->matchFunc = matchFunc;
    sm->refineFunc = refineFunc;

    // Preallocate CUDA memory for TVR
    cudaCalloc((void **) &sm->d_L, sizeof(float) * imgLen);
    cudaCalloc((void **) &sm->d_p, sizeof(float) * imgLen * 2);
    cudaCalloc((void **) &sm->d_q, sizeof(float) * imgLen * 4);
    cudaCalloc((void **) &sm->d_u, sizeof(float) * imgLen);
    cudaCalloc((void **) &sm->d_a, sizeof(float) * imgLen);
    cudaCalloc((void **) &sm->d_uHat, sizeof(float) * imgLen);
    cudaCalloc((void **) &sm->d_g, sizeof(float) * imgLen * 4);
    cudaCalloc((void **) &sm->d_leftNorm, sizeof(float) * imgLen);
    cudaCalloc((void **) &sm->d_grad, sizeof(float) * imgLen * 2);
    cudaCalloc((void **) &sm->d_gradBard, sizeof(float) * imgLen);
    cudaCalloc((void **) &sm->d_v, sizeof(float) * imgLen * 2);
    cudaCalloc((void **) &sm->d_vHat, sizeof(float) * imgLen * 2);
    cudaCalloc((void **) &sm->d_gradVHat, sizeof(float) * imgLen * 4);
    cudaCalloc((void **) &sm->d_gradUHat, sizeof(float) * imgLen * 2);
    cudaCalloc((void **) &sm->d_Gxp, sizeof(float) * imgLen * 2);

    return sm;
}

#define ABS_NORM_SUM(val) abs(val)
#define ABS_NORM_EXPONENT(val) abs(val)
#define EUCLID_NORM_SUM(val) (val) * (val)
#define EUCLID_NORM_EXPONENT(val) (val) * (val)

// // Normalise costs -> [0, 1]
// void normaliseCosts(StereoMatcher *sm) {
//   int width = sm->left->width;
//   int height = sm->left->height;
//   int range = sm->disparityRange;
//
//   for (int y = 0; y < height; y++) {
//     for (int x = 0; x < width; x++) {
//       double maxCost = COST_AT(x, y, 0, sm);
//
//       for (int d = 1; d < range; d++) {
//         double cost = COST_AT(x, y, d, sm);
//         if (cost > maxCost) maxCost = cost;
//       }
//
//       for (int d = 0; d < range; d++) {
//         if (COST_AT(x, y, d, sm) == -1) continue;
//         COST_AT(x, y, d, sm) /= maxCost;
//       }
//     }
//   }
// }

Image *genDisparityImage(StereoMatcher *sm) {
  struct timeval begin, end;

  if (sm->smoothFunc != NULL) {
    updateMsg("Smoothing");
    gettimeofday(&begin, 0);
    sm->smoothFunc(sm->left);
    sm->smoothFunc(sm->right);
    gettimeofday(&end, 0);
    double timeDiff = (1000000.0 * (end.tv_sec - begin.tv_sec) + end.tv_usec-begin.tv_usec) / 1000000.0;
    printf("- smoothing: %fs\n", timeDiff);
  }

  if (sm->matchFunc != NULL) {
    updateMsg("Computing matching costs");
    gettimeofday(&begin, 0);
    sm->matchFunc(sm);
    // normaliseCosts(sm);
    gettimeofday(&end, 0);
    double timeDiff = (1000000.0 * (end.tv_sec - begin.tv_sec) + end.tv_usec-begin.tv_usec) / 1000000.0;
    printf("- matching: %fs\n", timeDiff);
  }

  if (sm->aggregationFunc != NULL) {
    updateMsg("Aggregating costs");
    gettimeofday(&begin, 0);
    sm->aggregationFunc(sm);
    // normaliseCosts(sm);
    gettimeofday(&end, 0);
    double timeDiff = (1000000.0 * (end.tv_sec - begin.tv_sec) + end.tv_usec-begin.tv_usec) / 1000000.0;
    printf("- aggregation: %fs\n", timeDiff);
  }

  updateMsg("Generating disparity image");

  // Disparity estimatoin/selection?
  if (sm->selectionFunc != NULL) {
    updateMsg("Selecting disparities");
    gettimeofday(&begin, 0);
    sm->selectionFunc(sm);
    gettimeofday(&end, 0);
    double timeDiff = (1000000.0 * (end.tv_sec - begin.tv_sec) + end.tv_usec-begin.tv_usec) / 1000000.0;
    printf("- selection: %fs\n", timeDiff);
  }

  if (sm->refineFunc != NULL) {
    updateMsg("Peforming disparity refinement");
    gettimeofday(&begin, 0);
    sm->refineFunc(sm);
    gettimeofday(&end, 0);
    double timeDiff = (1000000.0 * (end.tv_sec - begin.tv_sec) + end.tv_usec-begin.tv_usec) / 1000000.0;
    printf("- refine: %fs\n", timeDiff);
  }

  return sm->disparityImg;
}
