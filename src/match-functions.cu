#include "match-functions.h"
#include "selection-functions.h"
#include "imageio.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#define TPB 512

#define BLOCK_RADIUS 2
#define BLOCK_SIZE (2 * BLOCK_RADIUS + 1)
#define BLOCK_LENGTH (BLOCK_SIZE * BLOCK_SIZE)

#define CENSUS_BLOCK_RADIUS 3
#define CENSUS_BLOCK_SIZE (2 * CENSUS_BLOCK_RADIUS + 1)
#define CENSUS_BLOCK_LENGTH (CENSUS_BLOCK_SIZE * CENSUS_BLOCK_SIZE)

#define BINARY_FEATURE_BLOCK_RADIUS 8
#define BINARY_FEATURE_BLOCK_SIZE (2 * BINARY_FEATURE_BLOCK_RADIUS + 1)
#define BINARY_FEATURE_BLOCK_LENGTH (CENSUS_BLOCK_SIZE * CENSUS_BLOCK_SIZE)


#define BLOCK_DIFF(x1, y1, img1, x2, y2, img2, diff, transform) \
  for (int _bdiff_x = -BLOCK_RADIUS; _bdiff_x <= BLOCK_RADIUS; _bdiff_x++) { \
    for (int _bdiff_y = -BLOCK_RADIUS; _bdiff_y <= BLOCK_RADIUS; _bdiff_y++) { \
      diff += transform((SAFE_PIXEL_AT((x1) + _bdiff_x, (y1) + _bdiff_y, img1, 0) - \
        SAFE_PIXEL_AT((x2) + _bdiff_x, (y2) + _bdiff_y, img2, 0))); \
    } \
  }

#define BLOCK_DIFF_CUDA(x1, y1, img1, x2, y2, img2, dimX, dimY, dimZ, diff, transform) \
  for (int _bdiff_x = -BLOCK_RADIUS; _bdiff_x <= BLOCK_RADIUS; _bdiff_x++) { \
    for (int _bdiff_y = -BLOCK_RADIUS; _bdiff_y <= BLOCK_RADIUS; _bdiff_y++) { \
      diff += transform((SAFE_PIXEL_AT_CUDA((x1) + _bdiff_x, (y1) + _bdiff_y, dimX, dimY, img1, 0) - \
        SAFE_PIXEL_AT_CUDA((x2) + _bdiff_x, (y2) + _bdiff_y, dimX, dimY, img2, 0))); \
    } \
  }

float square(float i) {
  return i * i;
}


__global__ void censusGen(
  const int *pixels, int64_t *censusVals,
  int width, int height, int imgLen) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= imgLen) return;

  int x = i % width;
  int y = i / width;

  float intensity = PIXEL_AT_INDEX_CUDA(i, pixels);

  int64_t census = 0;

  for (int yDiff = -CENSUS_BLOCK_RADIUS; yDiff <= CENSUS_BLOCK_RADIUS; yDiff++) {
    for (int xDiff = -CENSUS_BLOCK_RADIUS; xDiff <= CENSUS_BLOCK_RADIUS; xDiff++) {
      if (!yDiff && !xDiff) continue;
      // if (yDiff > 0) continue;

      census <<= 1;
      census |= SAFE_PIXEL_AT_CUDA(x + xDiff, y + yDiff,
        width, height, pixels, 0) < intensity;
    }
  }

  censusVals[i] = census;
}

__global__ void binaryGen(
  const int *pixels, int64_t *featureVals, const int *featureExtractor,
  int width, int height, int imgLen) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > imgLen) return;

  int x = i % width;
  int y = i / width;

  int64_t feature = 0;
  const int *feCH1 = featureExtractor;
  const int *feCH2 = feCH1 + FEATURE_EXTRACTOR_ITEMS;

  for (int i = 0; i < FEATURE_EXTRACTOR_ITEMS; i++) {
    int loc1 = feCH1[i];
    int loc2 = feCH2[i];

    int xDiff1 = loc1 % BINARY_FEATURE_BLOCK_SIZE - BINARY_FEATURE_BLOCK_RADIUS;
    int xDiff2 = loc2 % BINARY_FEATURE_BLOCK_SIZE - BINARY_FEATURE_BLOCK_RADIUS;

    int yDiff1 = loc1 / BINARY_FEATURE_BLOCK_SIZE - BINARY_FEATURE_BLOCK_RADIUS;
    int yDiff2 = loc2 / BINARY_FEATURE_BLOCK_SIZE - BINARY_FEATURE_BLOCK_RADIUS;

    feature <<= 1;
    feature |= SAFE_PIXEL_AT_CUDA(x + xDiff1, y + yDiff1,
      width, height, pixels, 0) < SAFE_PIXEL_AT_CUDA(x + xDiff2, y + yDiff2,
        width, height, pixels, 0);
  }

  featureVals[i] = feature;
}

__global__ void binaryReduce(int64_t *censusValsMatching,
  int64_t *censusValsReference, float *matchingCosts,
  int width, int range, int disparityLen) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i >= disparityLen) return;

  int pos = i / range;
  int x = pos % width;
  int y = pos / width;
  int d = i % range;

  int difference = -1;

  if (x - d >= 0) {
    difference = 0;
    int64_t xorVal = censusValsMatching[x + y * width] ^
      censusValsReference[x + y * width - d];

    // Count the number of bits that are set to 1 in a 64 bit integer
    difference = __popcll(xorVal);
  }

  COST_AT_CUDA(x, y, d, width, range, matchingCosts) = difference;
}

void CU_census(StereoMatcher *self) {
  int width = self->left->width;
  int height = self->left->height;
  int imgLen = width * height;
  int range = self->disparityRange;
  int disparityLen = imgLen * range;

  Image *matchingImage = self->left;
  Image *referenceImage = self->right;

  int64_t *censusValsMatching;
  cudaMalloc((void **) &censusValsMatching, sizeof(int64_t) * imgLen);

  int64_t *censusValsReference;
  cudaMalloc((void **) &censusValsReference, sizeof(int64_t) * imgLen);

  censusGen<<<ceil(1.0 * imgLen / TPB), TPB>>>(matchingImage->pixels,
    censusValsMatching, width, height, imgLen);

  censusGen<<<ceil(1.0 * imgLen / TPB), TPB>>>(referenceImage->pixels,
    censusValsReference, width, height, imgLen);

  cudaDeviceSynchronize();

  binaryReduce<<<ceil(1.0 * disparityLen / TPB), TPB>>>(censusValsMatching,
    censusValsReference, self->matchingCostsRead, width, range, disparityLen);

  cudaDeviceSynchronize();

  CU_WTA(self);

  saveImg("stage0000.pgm", self->disparityImg);
}

void CU_binaryFilter(StereoMatcher *self) {
  int width = self->left->width;
  int height = self->left->height;
  int imgLen = width * height;
  int range = self->disparityRange;
  int disparityLen = imgLen * range;

  Image *matchingImage = self->left;
  Image *referenceImage = self->right;

  int64_t *censusValsMatching;
  cudaMalloc((void **) &censusValsMatching, sizeof(int64_t) * imgLen);

  int64_t *censusValsReference;
  cudaMalloc((void **) &censusValsReference, sizeof(int64_t) * imgLen);

  binaryGen<<<ceil(1.0 * imgLen / TPB), TPB>>>(matchingImage->pixels,
    censusValsMatching, self->d_fe, width, height, imgLen);

  binaryGen<<<ceil(1.0 * imgLen / TPB), TPB>>>(referenceImage->pixels,
    censusValsReference, self->d_fe, width, height, imgLen);

  cudaDeviceSynchronize();

  binaryReduce<<<ceil(1.0 * disparityLen / TPB), TPB>>>(censusValsMatching,
    censusValsReference, self->matchingCostsRead, width, range, disparityLen);

  cudaDeviceSynchronize();

  CU_WTA(self);

  saveImg("stage0000.pgm", self->disparityImg);
}


// void census(StereoMatcher *self) {
//   int width = self->left->width;
//   int height = self->left->height;
//   int range = self->disparityRange;
//
//   Image *matchingImage = self->left;
//   Image *referenceImage = self->right;
//
//   for (int y = 0; y < height; y++) {
//     for (int x = 0; x < width; x++) {
//       float intensity = PIXEL_AT(x, y, matchingImage);
//
//       int64_t census = 0;
//
//       #pragma unroll CENSUS_BLOCK_SIZE
//       for (int yDiff = -CENSUS_BLOCK_RADIUS; yDiff <= CENSUS_BLOCK_RADIUS; yDiff++) {
//         #pragma unroll CENSUS_BLOCK_SIZE
//         for (int xDiff = -CENSUS_BLOCK_RADIUS; xDiff <= CENSUS_BLOCK_RADIUS; xDiff++) {
//           if (!xDiff && !yDiff) continue;
//
//           census <<= 1;
//           census |= SAFE_PIXEL_AT(x + xDiff, y + yDiff,
//             matchingImage, 0) < intensity;
//         }
//       }
//
//       for (int d = 0; d < range; d++) {
//         int difference = -1;
//
//         if (x - d >= 0) {
//           int64_t disparityCensus = 0;
//
//           #pragma unroll CENSUS_BLOCK_SIZE
//           for (int yDiff = -CENSUS_BLOCK_RADIUS; yDiff <= CENSUS_BLOCK_RADIUS; yDiff++) {
//             #pragma unroll CENSUS_BLOCK_SIZE
//             for (int xDiff = -CENSUS_BLOCK_RADIUS; xDiff <= CENSUS_BLOCK_RADIUS; xDiff++) {
//               if (!xDiff && !yDiff) continue;
//
//               disparityCensus <<= 1;
//               disparityCensus |= SAFE_PIXEL_AT(x - d + xDiff, y + yDiff,
//                 referenceImage, 0) < intensity;
//             }
//           }
//
//           /*int val = census ^ disparityCensus;
//
//           // Count the number of bits set
//           while (val != 0) {
//             // A bit is set, so increment the count and clear the bit
//             difference++;
//             val &= val - 1;
//           }*/
//
//           int xorVal = census ^ disparityCensus;
//
//           for (int i = 0; i < 4 * CENSUS_BLOCK_RADIUS * CENSUS_BLOCK_RADIUS - 1; i++) {
//             difference += xorVal & 1;
//             xorVal >>= 1;
//           }
//         }
//
//         COST_AT(x, y, d, self) = difference;
//       }
//     }
//   }
// }
//
// __global__ void CU_SAD(int *matchingPixels, int *referencePixels, float *matchingCosts) {
//   int x = blockIdx.x;
//   int y = blockIdx.y;
//   int d = blockIdx.z;
//
//   // matchingCosts[x] = 1;
//   //
//   int difference = -1;
//
//   if (x - d >= 0) {
//     BLOCK_DIFF_CUDA(
//       x, y, matchingPixels, // first image
//       x - d, y, referencePixels, // second image
//       gridDim.x, gridDim.y, gridDim.z, // block dimensions
//       difference, abs // difference variable, transformation
//     )
//
//     // int x1 = x;
//     // int x2 = x - d;
//     // int y1 = y;
//     // int y2 = y;
//     //
//     // difference = (int) fabsf(1.0f * (matchingPixels[x1 + y2 * gridDim.x] - referencePixels[x2 + y2 * gridDim.x]));
//     //
//     //
//     // // for (int _bdiff_x = -BLOCK_RADIUS; _bdiff_x <= BLOCK_RADIUS; _bdiff_x++) {
//     // //   for (int _bdiff_y = -BLOCK_RADIUS; _bdiff_y <= BLOCK_RADIUS; _bdiff_y++) {
//     // //     // difference = abs((SAFE_PIXEL_AT_CUDA((x1) + _bdiff_x, (y1) + _bdiff_y, gridDim.x, gridDim.y, matchingPixels, 0) -
//     // //     //   SAFE_PIXEL_AT_CUDA((x2) + _bdiff_x, (y2) + _bdiff_y, gridDim.x, gridDim.y, referencePixels, 0)));
//     // //   }
//     // // }
//   }
//
//   COST_AT_CUDA(x, y, d, gridDim.x, gridDim.z, matchingCosts) = difference;
//
//   // printf("tId.x:%d tId.y:%d tId.z:%d tId.x:%d \n", tid);
//
//   // matchingCosts[0] = 5.0f;
//   // matchingCosts[1] = 5.0f;
//   // matchingCosts[2] = 5.0f;
//   // matchingCosts[3] = 5.0f;
//   // matchingCosts[4] = 5.0f;
//
// }
//
// void CU_SAD(StereoMatcher *self) {
//   int width = self->left->width;
//   int height = self->left->height;
//   int range = self->disparityRange;
//
//   int *matchingPixels = self->left->pixels;
//   int *referencePixels = self->right->pixels;
//
//   int *d_matchingPixels;
//   int *d_referencePixels;
//   float *d_matchingCosts;
//
//   cudaMalloc((void **) &d_matchingPixels, sizeof(int) * width * height);
//   cudaMalloc((void **) &d_referencePixels, sizeof(int) * width * height);
//   cudaMalloc((void **) &d_matchingCosts, sizeof(float) * width * height * range);
//
//   // Copy images to the device
//   cudaMemcpy(d_matchingPixels, matchingPixels, sizeof(int) * width * height, cudaMemcpyHostToDevice);
//   cudaMemcpy(d_referencePixels, referencePixels, sizeof(int) * width * height, cudaMemcpyHostToDevice);
//
//   dim3 grid(width, height, range);
//   dim3 block(1, 1, 1);
//   CU_SAD<<<grid, block>>>(d_matchingPixels, d_referencePixels, d_matchingCosts);
//
//   cudaMemcpy(self->matchingCosts, d_matchingCosts, sizeof(float) * width * height * range, cudaMemcpyDeviceToHost);
//
//   // Cleanup
//   cudaFree(d_matchingPixels);
//   cudaFree(d_referencePixels);
//   cudaFree(d_matchingCosts);
// }
//
// void SAD(StereoMatcher *self) {
//   int width = self->left->width;
//   int height = self->left->height;
//   int range = self->disparityRange;
//
//   Image *matchingImage = self->left;
//   Image *referenceImage = self->right;
//
//   for (int y = 0; y < height; y++) {
//     for (int x = 0; x < width; x++) {
//       for (int d = 0; d < range; d++) {
//         int difference = -1;
//
//         if (x - d >= 0) {
//           BLOCK_DIFF(
//             x, y, matchingImage, // first image
//             x - d, y, referenceImage, // second image
//             difference, abs // difference variable, transformation
//           )
//         }
//
//         COST_AT(x, y, d, self) = difference;
//       }
//     }
//   }
// }
//
// void SSD(StereoMatcher *self) {
//   int width = self->left->width;
//   int height = self->left->height;
//   int range = self->disparityRange;
//
//   Image *matchingImage = self->left;
//   Image *referenceImage = self->right;
//
//   float difference;
//   for (int y = 0; y < height; y++) {
//     for (int x = 0; x < width; x++) {
//       for (int d = 0; d < range; d++) {
//         difference = -1;
//
//         if (x - d >= 0) {
//           BLOCK_DIFF(
//             x, y, matchingImage, // first image
//             x - d, y, referenceImage, // second image
//             difference, square  // difference variable, transformation
//           )
//         }
//
//         COST_AT(x, y, d, self) = difference;
//       }
//     }
//   }
// }
