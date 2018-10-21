#include "imageio.h"

#include <stdio.h>
#include <stdlib.h>

Image *allocImg(int width, int height)
{
  Image *img = (Image *) malloc(sizeof(Image));

  int w = width;
  int h = height;

  img->width = w;
  img->height = h;
  // img->pixels = (int *) malloc(w * h, sizeof(int));

  // printf("yeah\n");
  // int *pixelsTmp;
  cudaMalloc((void **) &img->pixels, sizeof(int) * w * h);
  // img->pixels = pixelsTmp;
  // printf("nah\n");

  // cudaMalloc((void **) &img->pixels, sizeof(int) * w * h);
  //
  // img->pixelsHost = new int[w * h];

  return img;
}

Image *loadImg(const char *filename)
{
  //Try to open the file
  FILE *fp = fopen(filename, "r");

  if (fp == 0) {
    return 0;
  }

  //Check the magic number for PGM files
  if (getc(fp) != 'P' || getc(fp) != '5') {
    fclose(fp);
    return 0;
  }

  while (getc(fp) != '\n') {}

  while (getc(fp) == '#') {
    while (getc(fp) != '\n') {}
  }


  fseek(fp, -1, SEEK_CUR);

  //Get the dimensions
  int width;
  int height;
  int maxval;
  if (fscanf(fp, "%d", &width) != 1 || fscanf(fp, "%d", &height) != 1 || fscanf(fp, "%d", &maxval) != 1) {
    fclose(fp);
    return 0;
  }

  printf("%d, %d\n", width, height);

  // Obsiously need to pad the image to allow for overflow
  Image *img = allocImg(width, height);
  int *pixelsTmp = (int *) malloc(img->width * img->height * sizeof(int));

  // printf("%p\n", (void *) &pixelsTmp);

  //Read in the pixel values
  unsigned char *buf = (unsigned char *) malloc(width);

  //Rows are stored in reverse order for this file format
  for (int r = height - 1; r >= 0; r--) {
    //Read in a byte for each pixel
    if (fread(buf, 1, width, fp) != width) {
      fclose(fp);
      // free(pixelsTmp);
      free(buf);
      freeImg(img);
      return 0;
    }

    for (int i = 0; i < width; i++) {
        // handle padding
        pixelsTmp[r * width + i] = (int) buf[i];

        // int buffTmp = (int) buf[i];
        //
        // cudaMemcpy(img->pixels + r * width + i, &buffTmp,
        //   sizeof(int), cudaMemcpyHostToDevice);
    }
  }

  cudaMemcpy(img->pixels, pixelsTmp,
    sizeof(int) * width * height, cudaMemcpyHostToDevice);
  //
  // cudaDeviceSynchronize();

  free(pixelsTmp);
  free(buf);
  fclose(fp);

  return img;
}

void saveImg(const char *filename, Image *img) {
  saveImg(filename, img, 1);
}

void saveImg(const char *filename, Image *img, int disparityScale) {
  FILE *fp = fopen(filename, "w");

  fprintf(fp, "P5\n");
  fprintf(fp, "%d %d\n%d\n", img->width, img->height, 255);

  unsigned char *buf = (unsigned char *) malloc(img->width);

  int *pixels = (int *) calloc(img->width * img->height, sizeof(int));
  cudaMemcpy(pixels, img->pixels,
    sizeof(int) * img->width * img->height, cudaMemcpyDeviceToHost);

  for (int r = img->height - 1; r >= 0; r--) {
    for (int i = 0; i < img->width; i++) {
      buf[i] = (unsigned char) pixels[r * img->width + i] * disparityScale;
    }

    fwrite(buf, 1, img->width, fp);
  }

  free(pixels);
  free(buf);
  fclose(fp);
}

void freeImg(Image *img) {
  cudaFree(img->pixels);
  free(img);
}
