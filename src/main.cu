#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include "imageio.h"
#include "stereo-matching.h"
#include "match-functions.h"
#include "aggregation-functions.h"
#include "selection-functions.h"
#include "smooth-functions.h"
#include "refine-functions.h"
#include "logger.h"

int main(int argc, char** argv) {

  if (argc != 7) {
    printf("Usage: ./matching <left image> <right image>"
      " <out image> <encoded binary matcher> <disparity range> <disparity scale>\n");
    return EXIT_FAILURE;
  }

  char *leftImgName = argv[1];
  char *rightImgName = argv[2];
  char *disparityMapFileName = argv[3];
  char *binaryFilter = argv[4];
  int disparityRange = atoi(argv[5]);
  int disparityScale = atoi(argv[6]);

  printf("%d, %d\n", disparityRange, disparityScale);

  updateMsg("Loading left image");

  // Load an image (in pgm format) specified by the first command line argument
  Image *leftImg = loadImg(leftImgName);

  updateMsg("Loading right image");

  // Load an image (in pgm format) specified by the second command line argument
  Image *rightImg = loadImg(rightImgName);

  printf("WORKING GEEE\n");

  // Ensure input images have same dimensions
  if (leftImg->width != rightImg->width ||
    leftImg->height != rightImg->height) {
    printf("Input images need to have the same dimensions");
    return EXIT_FAILURE;
  }

  updateMsg("Setting up stereo matching");

  StereoMatcher *myMatcher = allocMatcher(
    leftImg, // Left image
    rightImg, // Right image
    disparityRange, // Disparity range
    binaryFilter,
    NULL, // Smoothing algorithm
    CU_TVR, // TVR, // Selection func
    CU_localAdaptiveSupportWeight, // CU_localAdaptiveSupportWeight, // Aggregation func
    CU_binaryFilter, // Matching cost algorithm
    NULL // MedianFiltering // Disparity refinement algorithm
  );

  struct timeval begin, end;
  gettimeofday(&begin, 0);
  Image *disparityImg = genDisparityImage(myMatcher);
  gettimeofday(&end, 0);
  double timeDiff = (1000000.0 * (end.tv_sec - begin.tv_sec) + end.tv_usec-begin.tv_usec) / 1000000.0;
  printf("- total: %fs\n", timeDiff);

  updateMsg("Saving disparity image");

  // for (int i = 0; i < disparityImg->width * disparityImg->height; i++) {
  //   disparityImg->pixels[i] *= disparityScale;
  // }

  saveImg(disparityMapFileName, disparityImg, disparityScale);

  return EXIT_SUCCESS;
}
