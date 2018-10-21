// #include <stdlib.h>
#include <stdio.h>
#include "aggregation-functions.h"
#include "selection-functions.h"
#include "imageio.h"
#include <math.h>

#define TPB 512

// Proportionality constant k
#define LASW_K 5

// Based on the strength of grouping by color similarity. Probs different since
// just using intensity?
//
#define GAMMA_C 14

// Determined empirically
#define GAMMA_P 8

#define AGGREGATION_BLOCK_RADIUS 3
#define AGGREGATION_BLOCK_SIZE (AGGREGATION_BLOCK_RADIUS * 2 + 1)
#define AGGREGATION_BLOCK_LENGTH (AGGREGATION_BLOCK_SIZE * AGGREGATION_BLOCK_SIZE)

__device__ float w(int pX, int pY, int qX, int qY, int width, int height, int *pixels) {
  // Delta c - calculate intensity difference?
  int p = SAFE_PIXEL_AT_CUDA(pX, pY, width, height, pixels, 0);
  int q = SAFE_PIXEL_AT_CUDA(qX, qY, width, height, pixels, 0);
  float deltaC = abs(p - q);

  // double deltaC = sqrtf(p * p + q * q);

  // float p_l, p_a, p_b;
  // rgb2lab(p, p, p, &p_l, &p_a, &p_b);
  //
  // float q_l, q_a, q_b;
  // rgb2lab(q, q, q, &q_l, &q_a, &q_b);
  //
  // float l = p_l - q_l;
  // float a = p_a - q_a;
  // float b = p_b - q_b;
  //
  // float deltaC = sqrtf(l * l + a * a + b * b);

  // Delta g - calculate intensity difference?
  int diffX = pX - qX;
  int diffY = pY - qY;
  float deltaG = sqrtf(diffX * diffX + diffY * diffY);

  return LASW_K * __expf(-(deltaC / GAMMA_C  + deltaG / GAMMA_P));
}

__host__ float w(int pX, int pY, int qX, int qY, Image *img) {
  // Delta c - calculate intensity difference?
  int p = SAFE_PIXEL_AT(pX, pY, img, 0);
  int q = SAFE_PIXEL_AT(qX, qY, img, 0);
  float deltaC = sqrt(p * p + q * q);

  // Delta g - calculate intensity difference?
  int diffX = pX - qX;
  int diffY = pY - qY;
  float deltaG = sqrt(diffX * diffX + diffY * diffY);

  return LASW_K * exp(-(deltaC / GAMMA_C + deltaG / GAMMA_P));
}

__global__ void LASW_aggregate(
  int *matchingPixels, int *referencePixels,
  float *matchingCostsIn, float *matchingCostsOut,
  int width, int height, int range, int disparityLen) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= disparityLen) return;

  int pos = i / range;
  int x = pos % width;
  int y = pos / width;
  int d = i % range;

  int pX = x;
  int pY = y;

  int pBarX = x - d;
  int pBarY = y;

  if (pBarX < 0) {
    matchingCostsOut[i] =
      matchingCostsIn[i];
    return;
  }

  float upperSum = 0;
  float lowerSum = 0;

  for (int yDiff = -AGGREGATION_BLOCK_RADIUS; yDiff <= AGGREGATION_BLOCK_RADIUS; yDiff++) {
    for (int xDiff = -AGGREGATION_BLOCK_RADIUS; xDiff <= AGGREGATION_BLOCK_RADIUS; xDiff++) {
      if (!xDiff && !yDiff) continue;

      int qX = pX + xDiff;
      int qY = pY + yDiff;

      int qBarX = pBarX + xDiff;
      int qBarY = pBarY + yDiff;

      float wPQxwPQBar = w(pX, pY, qX, qY, width, height, matchingPixels) *
        w(pBarX, pBarY, qBarX, qBarY, width, height, referencePixels);

      float costVal = SAFE_COST_AT_CUDA(qX, qY, d,
        width, height, range, matchingCostsIn, 1);

      if (costVal == -1) {
        continue;
        costVal = 1;
      }

      upperSum += wPQxwPQBar * SAFE_COST_AT_CUDA(qX, qY, d,
        width, height, range, matchingCostsIn, 1);

      lowerSum += wPQxwPQBar;
    }
  }

  matchingCostsOut[i] = upperSum / lowerSum;
}

void CU_localAdaptiveSupportWeight(StereoMatcher *self) {
  int width = self->left->width;
  int height = self->left->height;
  int range = self->disparityRange;
  int imgLen = width * height;
  int disparityLen = imgLen * range;

  Image *matchingImage = self->left;
  Image *referenceImage = self->right;

  LASW_aggregate<<<ceil(1.0 * disparityLen / TPB), TPB>>>(
    matchingImage->pixels, referenceImage->pixels,
    self->matchingCostsRead, self->matchingCostsWrite,
    width, height, range, disparityLen);

  cudaDeviceSynchronize();

  float *tmpRead = self->matchingCostsRead;
  self->matchingCostsRead = self->matchingCostsWrite;
  self->matchingCostsWrite = tmpRead;

  CU_WTA(self);

  // CU_WTA(self);
  //
  // saveImg("stage0001.pgm", self->disparityImg);
}

// void localAdaptiveSupportWeight(StereoMatcher *self) {
//   int width = self->left->width;
//   int height = self->left->height;
//   int range = self->disparityRange;
//   int disparityLength = width * height * range;
//
//   Image *matchingImage = self->left;
//   Image *referenceImage = self->right;
//
//   StereoMatcher *aggregationCosts = allocMatcher(
//     self->left,
//     self->right,
//     self->disparityRange,
//     NULL, NULL, NULL, NULL, NULL
//   );
//
//   int aggregationBlockRadius = 3;
//   int progress = 0;
//
//   for (int y = 0; y < height; y++) {
//     for (int x = 0; x < width; x++) {
//       int pX = x;
//       int pY = y;
//
//       for (int d = 0; d < range; d++) {
//         progress += 1;
//
//         if (!(progress % (int) (disparityLength * 0.05))) {
//           printf("- aggregation %.0f%% complete\n", 100.0f * progress / disparityLength);
//         }
//
//         int pBarX = x - d;
//         int pBarY = y;
//
//         if (pBarX >= 0) {
//
//           float upperSum = 0;
//           float lowerSum = 0;
//
//           for (int yDiff = -aggregationBlockRadius; yDiff <= aggregationBlockRadius; yDiff++) {
//             for (int xDiff = -aggregationBlockRadius; xDiff <= aggregationBlockRadius; xDiff++) {
//               if (!xDiff && !yDiff) continue;
//
//               int qX = pX + xDiff;
//               int qY = pY + yDiff;
//
//               int qBarX = pBarX + xDiff;
//               int qBarY = pBarY + yDiff;
//
//               float wPQxwPQBar = w(pX, pY, qX, qY, matchingImage) *
//                 w(pBarX, pBarY, qBarX, qBarY, referenceImage);
//
//               float costVal = SAFE_COST_AT(qX, qY, d, self, 1);
//
//               if (costVal == -1) {
//                 continue;
//                 costVal = 1;
//               }
//
//               upperSum += wPQxwPQBar * SAFE_COST_AT(qX, qY, d, self, 1);
//
//               lowerSum += wPQxwPQBar;
//             }
//           }
//
//           COST_AT(x, y, d, aggregationCosts) = upperSum / lowerSum;
//         } else {
//           COST_AT(x, y, d, aggregationCosts) = COST_AT(x, y, d, self);
//         }
//       }
//     }
//   }
//
//   for (int i = 0; i < disparityLength; i++) {
//     self->matchingCosts[i] = aggregationCosts->matchingCosts[i];
//   }
// }
