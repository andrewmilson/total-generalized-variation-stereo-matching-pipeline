#include <math.h>
#include <stdlib.h>
#include "smooth-functions.h"

#define KERNEL_RADIUS 2
#define KERNEL_SIZE (2 * KERNEL_RADIUS + 1)
#define KERNEL_LENGTH (KERNEL_SIZE * KERNEL_SIZE)

void gaussian(Image *img) {
  int width = img->width;
  int height = img->height;
  Image *tmpImg = allocImg(width, height);
  Image *destImg = allocImg(width, height);

  double *kernel = (double *) malloc(KERNEL_SIZE * sizeof(double));

  // sigma determines the "spread" of the "Gaussian bell"
  double sigma = KERNEL_RADIUS / 2.0;

  // Calc kernels and normalize them
  double val;
  double sum = 0;
  for (int i = 0; i < KERNEL_SIZE; ++i) {
      val = abs(KERNEL_RADIUS - i);

      kernel[i] = (1 / sqrt(2 * M_PI * sigma * sigma)) *
        exp(-(val * val) / (2 * sigma * sigma));

      sum += kernel[i];
  }

  // Normalize
  for (int i = 0; i < KERNEL_SIZE; ++i) {
      kernel[i] /= sum;
  }

  // Convolve image with our kernel
  double valueX;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      valueX = 0;

      for (int k = 0; k < KERNEL_SIZE; ++k) {
        valueX += kernel[k] * SAFE_PIXEL_AT(x + k - KERNEL_RADIUS, y, img, 0);
      }

      PIXEL_AT(x, y, tmpImg) = valueX;
    }
  }

  double valueY;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      valueY = 0;

      for (int k = 0; k < KERNEL_SIZE; ++k) {
        valueY += kernel[k] * SAFE_PIXEL_AT(x, y + k - KERNEL_RADIUS, tmpImg, 0);
      }

      PIXEL_AT(x, y, destImg) = valueY;
    }
  }

  free(kernel);
  img->pixels = destImg->pixels;
  freeImg(tmpImg);
  free(destImg);
}
