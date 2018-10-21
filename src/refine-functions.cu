// #include <stdlib.h>
#include "refine-functions.h"
#include "imageio.h"
#include <stdlib.h>

#define NHOOD_LENGTH (3 * 3)

int cmpNumerical(const void *a, const void *b) {
  return (*(int *)a - *(int *)b);
}

void MedianFiltering(StereoMatcher *self) {
  Image *disparityImg = self->disparityImg;

  int width = disparityImg->width;
  int height = disparityImg->height;

  Image *tmpDisparity = allocImg(width, height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      // NW | NC | NE
      // ------------
      // CW | CC | CE
      // ------------
      // SW | SC | SE

      int NC = x + (y - 1) * width;
      int NE = x + 1 + (y + 1) * width;
      int CE = x + 1 + y * width;
      int SE = x + 1 + (y + 1) * width;
      int SC = x + (y + 1) * width;
      int SW = x - 1 + (y + 1) * width;
      int CW = x - 1 + y * width;
      int NW = x - 1 + (y - 1) * width;
      int CC = x + y * width;


      int nhoodVals[NHOOD_LENGTH] = {
        SAFE_PIXEL_AT_INDEX(NW, disparityImg, -1), SAFE_PIXEL_AT_INDEX(NC, disparityImg, -1), SAFE_PIXEL_AT_INDEX(NE, disparityImg, -1),
        SAFE_PIXEL_AT_INDEX(CW, disparityImg, -1), SAFE_PIXEL_AT_INDEX(CC, disparityImg, -1), SAFE_PIXEL_AT_INDEX(CE, disparityImg, -1),
        SAFE_PIXEL_AT_INDEX(SW, disparityImg, -1), SAFE_PIXEL_AT_INDEX(SC, disparityImg, -1), SAFE_PIXEL_AT_INDEX(SE, disparityImg, -1)
      };

      qsort(nhoodVals, NHOOD_LENGTH, sizeof(int), cmpNumerical);

      int medianStartPos = -1;

      for (int i = 0; i < NHOOD_LENGTH; i++) {
        if (nhoodVals[i] != -1) {
          medianStartPos = i;
          break;
        }
      }

      int trueLength = NHOOD_LENGTH - medianStartPos;

      double median;

      if (trueLength % 2 == 0) {
        median = (nhoodVals[medianStartPos + trueLength / 2] +
          nhoodVals[medianStartPos + trueLength / 2 - 1]) / 2;
      } else {
        median = nhoodVals[medianStartPos + trueLength / 2];
      }

      // Replace disparity at x, y with calculated median
      PIXEL_AT(x, y, tmpDisparity) = median;
    }
  }

  // free(disparityImg->pixels);
  // disparityImg->pixels = tmpDisparity->pixels;
  // free(tmpDisparity);
  self->disparityImg = tmpDisparity;
}

void OcclusionFilling(StereoMatcher *self) {

}
