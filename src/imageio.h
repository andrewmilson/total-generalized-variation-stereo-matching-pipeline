#ifndef __IMAGEIO_H__
#define __IMAGEIO_H__

typedef struct
{
    int width;
    int height;
    int *pixels;
} Image;

Image *allocImg(int width, int height);
Image *loadImg(const char *filename);
void saveImg(const char *filename, Image *img);
void saveImg(const char *filename, Image *img, int disparityScale);
void freeImg(Image *img);

#define PIXEL_AT_INDEX(i, img) (img->pixels[i])
#define PIXEL_AT_INDEX_CUDA(i, img) (img[i])

#define PIXEL_AT(x, y, img) PIXEL_AT_INDEX(img->width * (y) + (x), img)
#define PIXEL_AT_CUDA(x, y, dimX, img) PIXEL_AT_INDEX_CUDA((dimX) * (y) + (x), img)

// Takes index into pixel array. Checks if there is a pixel at the given x and
// y, if not gives undefVal. Slower
#define SAFE_PIXEL_AT_INDEX(i, img, undefVal) ( \
  (i) <= img->width * img->height - 1 && \
  (i) >= 0 ? PIXEL_AT_INDEX(i, img) : (undefVal))
#define SAFE_PIXEL_AT_INDEX_CUDA(i, dimX, dimY, img, undefVal) (\
  (i) < (dimX) * (dimY) && (i) >= 0 ? PIXEL_AT_INDEX_CUDA(i, img) : (undefVal))

// Takes x and y. Checks if there is a pixel at the given x and y, if not gives
// undefVal. Slower
#define SAFE_PIXEL_AT(x, y, img, undefVal) \
  SAFE_PIXEL_AT_INDEX((x) + (y) * img->width, img, undefVal)
#define SAFE_PIXEL_AT_CUDA(x, y, dimX, dimY, img, undefVal) \
  SAFE_PIXEL_AT_INDEX_CUDA((x) + (y) * (dimX), dimX, dimY, img, undefVal)

#endif
