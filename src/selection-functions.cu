#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "selection-functions.h"
#include "imageio.h"
#include <sys/time.h>

#define TPB 512

inline __host__ __device__ float maxD(float a, float b) {
  return a > b ? a : b;
}

inline __host__ __device__ float minD(float a, float b) {
  return a < b ? a : b;
}

__global__ void CU_WTA(int *disparityPixels, float *matchingCosts, int disparityRange) {
  int x = blockIdx.x;
  int y = blockIdx.y;
  int d = threadIdx.x;

  __shared__ int cacheDispIndex[256];
  __shared__ float cacheDispErr[256];
  cacheDispIndex[d] = d;
  cacheDispErr[d] = COST_AT_CUDA(x, y, d, gridDim.x, disparityRange, matchingCosts);
  __syncthreads();

  unsigned int i = blockDim.x / 2;

  while (i != 0) {
    if (d < i) {
      float tmp1 = cacheDispErr[d];
      float tmp2 = cacheDispErr[d + i];

      if (tmp2 != -1 && tmp1 > tmp2) {
        cacheDispIndex[d] = cacheDispIndex[d + i];
        cacheDispErr[d] = tmp2;
      }
    }

    __syncthreads();
    i /= 2;
  }

  // If first thread
  if (d == 0) {
    PIXEL_AT_CUDA(x, y, gridDim.x, disparityPixels) = cacheDispIndex[0];
  }
}

void CU_WTA(StereoMatcher *self) {
  Image *disparityImg = self->disparityImg;

  int width = disparityImg->width;
  int height = disparityImg->height;
  int range = self->disparityRange;

  dim3 grid(width, height);
  dim3 block(self->disparityRange);
  CU_WTA<<<grid, block>>>(disparityImg->pixels, self->matchingCostsRead, range);
}

// void WTA(StereoMatcher *self) {
//   Image *disparityImg = self->disparityImg;
//   int width = disparityImg->width;
//   int height = disparityImg->height;
//
//
//   int dispX = 100;
//   int dispY = 100;
//   float *disparities = (float *) malloc(self->disparityRange * sizeof(float));
//
//   for (int i = 0; i < self->disparityRange; i++)
//     disparities[i] = COST_AT(dispX, dispY, i, self);
//
//   for (int y = 0; y < height; y++) {
//     for (int x = 0; x < width; x++) {
//       float minDisparityErr = COST_AT(x, y, 0, self);
//       int minDisparityIndex = 0;
//
//       for (int d = 1; d < self->disparityRange; d++) {
//         float disparityErr = COST_AT(x, y, d, self);
//
//         if (disparityErr != -1 && disparityErr < minDisparityErr) {
//           minDisparityErr = disparityErr;
//           minDisparityIndex = d;
//         }
//       }
//
//       PIXEL_AT(x, y, disparityImg) = minDisparityIndex;
//     }
//   }
// }


// MIDDLEBURRY
// // Paper mentions for Middleburry dataset use λd=1.0 and λs=0.2
// #define LAMBDA_S 0.2f
// #define LAMBDA_A (8.0f * LAMBDA_S) // "obtained the best results for λa=8λs"
// #define LAMBDA_D 1.0f
//
// // Gradient descent step sizes
// #define TAU_U 0.28867513459f // ≈ 1 / sqrt(12)
// #define TAU_P 0.28867513459f // ≈ 1 / sqrt(12)
// #define TAU_V 0.35355339059f // ≈ 1 / sqrt(8)
// #define TAU_Q 0.35355339059f // ≈ 1 / sqrt(8)

// KITTI
// Paper mentions for Middleburry dataset use λd=1.0 and λs=0.2
#define LAMBDA_S 1.0f
#define LAMBDA_A (8.0f * LAMBDA_S) // "obtained the best results for λa=8λs"
#define LAMBDA_D 0.4f

// Gradient descent step sizes
#define TAU_U 0.28867513459f // ≈ 1 / sqrt(12)
#define TAU_P 0.28867513459f // ≈ 1 / sqrt(12)
#define TAU_V 0.35355339059f // ≈ 1 / sqrt(8)
#define TAU_Q 0.35355339059f // ≈ 1 / sqrt(8)

// controls how fast the convex and non-convex solution are drawn together
#define BETA 0.001f

#define ITTERATION_LIMIT 80

// // Weights for "a" and "b" not defined in paper?
// #define WEIGHT_A 10
// #define WEIGHT_B 2

#define PRINT_ARR(name, i, a) printf("%s[%d:%d] = %lf, %lf, %lf, %lf\n", \
  name, i, i + 3, a[i], a[i+1], a[i+2], a[i+3])

// Computes derivative of channel at index "i" using backwards differences
__host__ __device__ void finiteBackwardCH(
  float *dstGradX, float *dstGradY, float *channel,
  int width, int index) {
  //      ------
  //      | NC |
  // -----------
  // | CW | CC |
  // -----------
  float CC = channel[index];
  float NC = 0;
  float CW = 0;

  if (index % width != 0) {
    CW = channel[index - 1];
  }

  if (index / width > 0) {
    NC = channel[index - width];
  }

  dstGradX[0] = CC - CW;
  dstGradY[0] = CC - NC;
}

// Computes derivative of channel at index "i" using forward differences
__host__ __device__ void finiteForwardCH(
  float *dstGradX, float *dstGradY, float *channel,
  int width, int height, int index) {
  // -----------
  // | CC | CE |
  // -----------
  // | CS |
  // ------
  float CC = channel[index];
  float CE = CC;
  float CS = CC;

  if ((index + 1) % width != 0) {
    CE = channel[index + 1];
  }

  if (index / width < height - 1) {
    CS = channel[index + width];
  }

  dstGradX[0] = CE - CC;
  dstGradY[0] = CS - CC;
}

inline __device__ float2 operator+(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}
inline __device__ float2 operator-(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator+=(float2 &a, float2 b) {
a.x += b.x;
a.y += b.y;
}

inline __host__ __device__ float2 operator*(float2 a, float b) {
  return make_float2(a.x * b, a.y * b);
}
inline __host__ __device__ float2 operator*(float b, float2 a) {
  return make_float2(b * a.x, b * a.y);
}

inline __host__ __device__ float4 operator*(float4 a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
inline __host__ __device__ float4 operator*(float b, float4 a) {
  return make_float4(b * a.x, b * a.y, b * a.z, b * a.w);
}

inline __host__ __device__ void operator+=(float4 &a, float4 b) {
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  a.w += b.w;
}
inline __host__ __device__ void operator*=(float4 &a, float b) {
  a.x *= b;
  a.y *= b;
  a.z *= b;
  a.w *= b;
}
inline __host__ __device__ void operator*=(float2 &a, float b) {
  a.x *= b;
  a.y *= b;
}

inline __host__ __device__ float2 operator*(float4 g, float2 v) {
  return make_float2(v.x * g.x + v.y * g.y, v.x * g.z + v.y * g.w);
}

__device__ float2 matmul(float4 g, float2 v) {
  return make_float2(v.x * g.x + v.y * g.y, v.x * g.z + v.y * g.w);
}

// Computes derivative of channel at index "i" using forward differences
inline __device__ void FF(float2 &dstGrad, float *chan, bool chk1, bool chk2, int w, int i) {
  // -----------
  // | CC | CE |
  // -----------
  // | CS |
  // ------

  float CC = chan[i];
  float CE = chk1 ? chan[i + 1] : CC;
  float CS = chk2 ? chan[i + w] : CC;

  dstGrad.x = CE - CC;
  dstGrad.y = CS - CC;
}

// Computes derivative of channel at index "i" using forward differences
inline __device__ void FF(float4 &dstGrad, float2 *chan, bool chk1, bool chk2, int w, int i) {
  // -----------
  // | CC | CE |
  // -----------
  // | CS |
  // ------

  float2 CC = chan[i];
  float2 CE = chk1 ? chan[i + 1] : CC;
  float2 CS = chk2 ? chan[i + w] : CC;

  dstGrad.x = CE.x - CC.x;
  dstGrad.y = CS.x - CC.x;
  dstGrad.z = CE.y - CC.y;
  dstGrad.w = CS.y - CC.y;
}


inline __device__ float divergence(float2 *chan, bool chk1, bool chk2, int w, int i) {
  float2 CC = chan[i];
  float CW = chk1 ? chan[i - 1].x : 0;
  float NC = chk2 ? chan[i - w].y : 0;
  return CC.x + CC.y - CW - NC;
}


inline __device__ float2 divergence(float4 *chan, bool chk1, bool chk2, int w, int i) {
  float4 CC = chan[i];
  float2 CW = chk1 ? ((float2 *) chan)[(i - 1) * 2] : make_float2(0, 0);
  float2 NC = chk2 ? ((float2 *) chan)[(i - w) * 2 + 1] : make_float2(0, 0);

  return make_float2(CC.x - CW.x + CC.z - NC.x, CC.y - CW.y + CC.w - NC.y);
}

inline __device__ void FB(float4 &dstGrad, float2 *chan, int w, int i) {
  //      ------
  //      | NC |
  // -----------
  // | CW | CC |
  // -----------
  float2 CC = chan[i];
  float2 NC = make_float2(0, 0);
  float2 CW = make_float2(0, 0);

  if (i % w != 0) {
    CW = chan[i - 1];
  }

  if (i / w > 0) {
    NC = chan[i - w];
  }

  dstGrad.x = CC.x - CW.x;
  dstGrad.y = CC.x - NC.x;
  dstGrad.z = CC.y - CW.y;
  dstGrad.w = CC.y - NC.y;
}

inline __device__ void FB(float4 &dstGrad1, float4 &dstGrad2, float4 *chan, int w, int i) {
  //      ------
  //      | NC |
  // -----------
  // | CW | CC |
  // -----------
  float4 CC = chan[i];
  float4 NC = make_float4(0, 0, 0, 0);
  float4 CW = make_float4(0, 0, 0, 0);

  if (i % w != 0) {
    CW = chan[i - 1];
  }

  if (i / w > 0) {
    NC = chan[i - w];
  }

  dstGrad1.x = CC.x - CW.x;
  dstGrad1.y = CC.x - NC.x;
  dstGrad1.z = CC.y - CW.y;
  dstGrad1.w = CC.y - NC.y;

  dstGrad2.x = CC.z - CW.z;
  dstGrad2.y = CC.z - NC.z;
  dstGrad2.z = CC.w - CW.w;
  dstGrad2.w = CC.w - NC.w;
}

__global__ void updateLoopTVR_stg1(
  float2 *p,
  float4 *q,
  float *uHat,
  float2 *vHat,
  int width, int height, int imgLen
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= imgLen) return;
  bool chk1 = (i + 1) % width != 0;
  bool chk2 = i / width < height - 1;

  float CC1 = uHat[i];

  float2 uHatGrad = make_float2(
    chk1 ? uHat[i + 1] - CC1 : 0,
    chk2 ? uHat[i + width] - CC1 : 0
  );

  // grad vHat
  float2 CC2 = vHat[i];
  float2 CE = chk1 ? vHat[i + 1] : CC2;
  float2 CS = chk2 ? vHat[i + width] : CC2;

  float4 vHatGrad = make_float4(
    CE.x - CC2.x,
    CS.x - CC2.x,
    CE.y - CC2.y,
    CS.y - CC2.y
  );

  vHatGrad.y += vHatGrad.z;
  vHatGrad.y *= 0.5f;
  vHatGrad.z = vHatGrad.y;

  // p^(n+1)
  p[i] += TAU_P * (uHatGrad - vHat[i]);

  // q^(n+1)
  q[i] += TAU_Q * vHatGrad;

  float pNorm = fmaxf(fabsf(p[i].x), fabsf(p[i].y));
  // Compute fast division
  float pScale = fminf(1.0f, __fdividef(LAMBDA_S, pNorm));
  p[i] *= pScale;

  float qNorm = fmaxf(fabsf(q[i].x), fmaxf(fabsf(q[i].y), fmaxf(fabsf(q[i].z), fabsf(q[i].w))));
  // Compute fast division
  float qScale = fminf(1.0f, __fdividef(LAMBDA_A, qNorm));
  q[i] *= qScale;
}

__global__ void updateLoopTVR_stg2(
  float augmentedStepSize,
  float *L,
  float2 *p,
  float4 *q,
  float *u, float *uHat,
  float *a,
  float2 *v,
  float2 *vHat,
  int width, int height, int imgLen
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= imgLen) return;
  bool chk1 = (i % width) != 0; // fmodf(i, width);
  bool chk2 = i / width > 0;

  float2 CC = p[i];
  float divergenceP = CC.x + CC.y - (chk1 ? p[i - 1].x : 0) - (chk2 ? p[i - width].y : 0) - L[i];

  // u^(n+1)
  float uTmp = u[i];

  // Clamp argument to [+0.0, 1.0].
  float newU = __saturatef(
    // fast division
    __fdividef(
      uTmp + TAU_U * divergenceP + augmentedStepSize * a[i],
      1 + augmentedStepSize
    )
  );

  u[i] = newU;

  // uHat^(n+1)
  uHat[i] = 2 * newU - uTmp;

  // v^(n+1)
  float2 vTmp = v[i];
  float2 qDivergence = p[i] + divergence(q, chk1, chk2, width, i);
  v[i] += TAU_V * qDivergence;
  vHat[i] = 2 * v[i] - vTmp;
}

// __global__ void aUpdate_stg1(
//   float *matchingCostsIn, float *matchingCostsOut, float *L, float *u,
//   float theta, int disparityRange, int disparityLen, int width
// ) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i >= disparityLen) return;
//
//   float mcVal = matchingCostsIn[i];
//
//   if (mcVal == -1) {
//     matchingCostsOut[i] = -1;
//     return;
//   }
//
//   int pos = i / disparityRange;
//   int d = i % disparityRange;
//
//   float a = 1.0f * d / disparityRange;
//   float diff = u[pos] - a;
//   matchingCostsOut[i] = LAMBDA_D * mcVal + L[pos] * diff + diff * diff / (2 * theta);
// }

__global__ void aUpdate_stg1(
  int *disparityPixels, float *matchingCostsIn, float *matchingCostsOut, float *L, float *u, float *a,
  float theta, int disparityRange, int width
) {

  int x = blockIdx.x;
  int y = blockIdx.y;
  int d = threadIdx.x;
  int pos = x + y * width;
  int i = pos * disparityRange + d;

  float mcVal = 1.0f * matchingCostsIn[i] / 64;

  __shared__ int cacheDispIndex[128];
  __shared__ float cacheDispErr[128];

  cacheDispIndex[d] = d;

  if (mcVal < 0) {
    cacheDispErr[d] = -1;
  } else {
    float diff = u[pos] - 1.0f * d / disparityRange;
    cacheDispErr[d] = LAMBDA_D * mcVal + L[pos] * diff + diff * diff / (2 * theta);
    // matchingCostsIn[i] = cacheDispErr[d];
  }

  __syncthreads();

  unsigned int index = blockDim.x / 2;

  while (index != 0) {
    if (d < index) {
      float tmp1 = cacheDispErr[d];
      float tmp2 = cacheDispErr[d + index];

      if (tmp2 != -1 && (tmp1 > tmp2 || tmp1 == -1)) {
        cacheDispIndex[d] = cacheDispIndex[d + index];
        cacheDispErr[d] = tmp2;
      }
    }

    __syncthreads();
    index /= 2;
  }

  // If first thread
  if (d == 0) {
    disparityPixels[pos] = cacheDispIndex[0];
    a[pos] = 1.0f * cacheDispIndex[0] / disparityRange;
    L[pos] = L[pos] + (u[pos] - a[pos]) / (2 * theta);
  }
}

// __global__ void tvr_genDispImg(
//   int *disparityPixels, float *a, float *L, float *u,
//   float theta, int disparityRange, int imgLen
// ) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i >= imgLen) return;
//   disparityPixels[i] *= 4;
// }

__global__ void tvr_init1(int *disparityPixels, float *u, float *a,
  float *uHat, float *leftNorm, int imgLen, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= imgLen) return;

  // Normalise disparityImage -> [0, 1]
  u[i] = a[i] = uHat[i] = 1.0f * disparityPixels[i] / range;
  leftNorm[i] = 1.0f * disparityPixels[i] / 255.0f;
}

__global__ void tvr_init2(float *leftNorm, float *gradXd, float *gradYd,
  float *gradBard, int imgLen, int width, int height) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > imgLen) return;

  finiteForwardCH(gradXd + i, gradYd + i, leftNorm, width, height, i);
  gradBard[i] = sqrt(gradXd[i] * gradXd[i] + gradYd[i] * gradYd[i]);
}

// Total Variation Regularisation
// http://openaccess.thecvf.com/content_iccv_workshops_2013/W21/papers/Kuschk_Fast_and_Accurate_2013_ICCV_paper.pdf
void CU_TVR(StereoMatcher *self) {
  // Perform WTA to get initial values for "u" and "a"
  CU_WTA(self);

  int width = self->disparityImg->width;
  int height = self->disparityImg->height;
  int range = self->disparityRange;
  int imgLen = width * height;
  int disparityLen = imgLen * range;

  float *d_pCH1 = self->d_p;
  float *d_pCH2 = self->d_p + imgLen;

  float *d_qCH1 = self->d_q;
  float *d_qCH2 = self->d_q + imgLen;
  float *d_qCH3 = self->d_q + imgLen * 2;
  float *d_qCH4 = self->d_q + imgLen * 3;

  float *d_gCH1 = self->d_g;
  float *d_gCH2 = self->d_g + imgLen;
  float *d_gCH3 = self->d_g + imgLen * 2;
  float *d_gCH4 = self->d_g + imgLen * 3;

  float *d_gradXd = self->d_grad;
  float *d_gradYd = self->d_grad + imgLen;

  float *d_vCH1 = self->d_v;
  float *d_vCH2 = self->d_v + imgLen;

  float *d_vHatCH1 = self->d_vHat;
  float *d_vHatCH2 = self->d_vHat + imgLen;

  float *d_gradXvHatCH1 = self->d_gradVHat;
  float *d_gradYvHatCH1 = self->d_gradVHat + imgLen;
  float *d_gradXvHatCH2 = self->d_gradVHat + imgLen * 2;
  float *d_gradYvHatCH2 = self->d_gradVHat + imgLen * 3;

  float *d_gradXuHat = self->d_gradUHat;
  float *d_gradYuHat = self->d_gradUHat + imgLen;

  float *d_GxpCH1 = self->d_Gxp;
  float *d_GxpCH2 = self->d_Gxp + imgLen;


  tvr_init1<<<ceil(1.0 * imgLen / TPB), TPB>>>(self->disparityImg->pixels,
    self->d_u, self->d_a, self->d_uHat, self->d_leftNorm, imgLen, range);

  cudaDeviceSynchronize();

  tvr_init2<<<ceil(1.0 * imgLen / TPB), TPB>>>(self->d_leftNorm, d_gradXd, d_gradYd,
    self->d_gradBard, imgLen, width, height);

  int count = 1;
  float theta = 1;

  dim3 grid(width * height);
  dim3 gridDisp(width, height);

  dim3 gridImg(width, height);
  dim3 blockRange(range);

  dim3 block(1, 1, 1);

  double timeDiffSmooth = 0;
  double timeDiffA = 0;

  struct timeval begin, end;

  // Perform gradient descent
  while(count < ITTERATION_LIMIT) {
    printf("- TV itteration %d/%d\n", count, ITTERATION_LIMIT);

    gettimeofday(&begin, 0);

    for (int i = 0; i < 150; i++) {
      updateLoopTVR_stg1<<<ceil(1.0 * imgLen / TPB), TPB>>>(
        (float2 *) self->d_p,
        (float4 *) self->d_q,
        self->d_uHat,
        (float2 *) self->d_vHat,
        width, height, imgLen
      );

      updateLoopTVR_stg2<<<ceil(1.0 * imgLen / TPB), TPB>>>(
        TAU_U / theta,
        self->d_L,
        (float2 *) self->d_p,
        (float4 *) self->d_q,
        self->d_u, self->d_uHat,
        self->d_a,
        (float2 *) self->d_v,
        (float2 *) self->d_vHat,
        width, height, imgLen
      );
    }

    cudaDeviceSynchronize();
    gettimeofday(&end, 0);

    timeDiffSmooth += (1000000.0 * (end.tv_sec - begin.tv_sec) + end.tv_usec-begin.tv_usec) / 1000.0;

    gettimeofday(&begin, 0);

    aUpdate_stg1<<<gridImg, blockRange>>>(
      self->disparityImg->pixels, self->matchingCostsRead, self->matchingCostsWrite, self->d_L, self->d_u, self->d_a,
      theta, range, width
    );

    // aUpdate_stg1<<<ceil(1.0 * disparityLen / TPB), TPB>>>(
    //   self->matchingCostsRead, self->matchingCostsWrite, self->d_L, self->d_u,
    //   theta, range, disparityLen, width
    // );

    // cudaDeviceSynchronize();
    //
    // float *tmpRead = self->matchingCostsRead;
    // self->matchingCostsRead = self->matchingCostsWrite;
    // self->matchingCostsWrite = tmpRead;
    //
    // CU_WTA<<<gridImg, blockRange>>>(
    //   self->disparityImg->pixels, self->matchingCostsRead,
    //   range);

    cudaDeviceSynchronize();

    gettimeofday(&end, 0);

    timeDiffA += (1000000.0 * (end.tv_sec - begin.tv_sec) + end.tv_usec - begin.tv_usec) / 1000.0;

    // timeDiffA += (double)(end - begin) / CLOCKS_PER_SEC;

    // char stageName[20];
    // sprintf(stageName, "stage%04d.pgm", count + 2);
    // saveImg(stageName, self->disparityImg);

    theta *= 1 - BETA * count;
    count += 1;
  }

  // tvr_genDispImg<<<ceil(1.0 * imgLen / TPB), TPB>>>(
  //   self->disparityImg->pixels, self->d_a, self->d_L, self->d_u,
  //   theta, range, imgLen
  // );

  cudaDeviceSynchronize();

  printf("- Computing smoothing time: %f ms\n", timeDiffSmooth / ITTERATION_LIMIT);
  printf("- Computing \"a\" time: %f ms\n", timeDiffA / ITTERATION_LIMIT);

  // cudaMemcpy(self->disparityImg->pixels, d_disparityPixels, sizeof(int) * imgLen, cudaMemcpyDeviceToHost);
  //
  // for (int i = 0; i < imgLen; i++) {
  //   self->disparityImg->pixels[i] /= 4;
  // }

  printf("finished\n");
}


// // Total Variation Regularisation
// // http://openaccess.thecvf.com/content_iccv_workshops_2013/W21/papers/Kuschk_Fast_and_Accurate_2013_ICCV_paper.pdf
// void TVR(StereoMatcher *self) {
//   // Perform WTA to get initial values for "u" and "a"
//   WTA(self);
//
//   int width = self->disparityImg->width;
//   int height = self->disparityImg->height;
//   int imgLen = width * height;
//
//   // Init L
//   float *L = (float *) calloc(imgLen, sizeof(float));
//
//   // Init "p" and "q"
//   float *p = (float *) calloc(imgLen * 2, sizeof(float));
//   float *pCH1 = (float *) (p);
//   float *pCH2 = (float *) (p + imgLen);
//
//   float *q = (float *) calloc(imgLen * 4, sizeof(float));
//   float *qCH1 = (float *) (q);
//   float *qCH2 = (float *) (q + imgLen);
//   float *qCH3 = (float *) (q + imgLen * 2);
//   float *qCH4 = (float *) (q + imgLen * 3);
//
//
//   // Normalise disparityImage -> [0, 1]
//   float *u = (float *) malloc(sizeof(float) * imgLen);
//   float *a = (float *) malloc(sizeof(float) * imgLen);
//   float *uHat = (float *) calloc(imgLen, sizeof(float));
//
//   for (int i = 0; i < imgLen; i++) {
//     // TODO: assume there is a disparity that is the max disparity i.e.
//     // disparityRange.
//     uHat[i] = a[i] = u[i] = 1.0f * self->disparityImg->pixels[i] / self->disparityRange;
//   }
//
//   int count = 0;
//
//   // // G identity matrix but may change so will use it in calculations
//   // float G[4] = {1, 0, 0, 1};
//   // float *GR1C1 = G;
//   // float *gCH2 i G + 1;
//   // float *gCH3 i G + 2;
//   // float *gCH4 = G + 3;
//
//   float *gCH1 = (float *) calloc(imgLen * 4, sizeof(float));
//   float *gCH2 = (float *) (gCH1 + imgLen);
//   float *gCH3 = (float *) (gCH1 + imgLen * 2);
//   float *gCH4 = (float *) (gCH1 + imgLen * 3);
//
//   float *leftNorm = (float *) calloc(imgLen, sizeof(float));
//
//   for (int i = 0; i < imgLen; i++) {
//     leftNorm[i] = 1.0f * self->disparityImg->pixels[i] / 255.0;
//   }
//
//
//   float *gradXd = (float *) calloc(imgLen * 2, sizeof(float));
//   float *gradYd = (float *) (gradXd + imgLen);
//   float *gradBard = (float *) calloc(imgLen, sizeof(float));
//
//   for (int i = 0; i < imgLen; i++) {
//     gCH1[i] = 1;
//     gCH4[i] = 1;
//     finiteForwardCH(gradXd + i, gradYd + i, leftNorm, width, height, i);
//     gradBard[i] = sqrt(gradXd[i] * gradXd[i] + gradYd[i] * gradYd[i]);
//   }
//
//   float *vCH1 = (float *) calloc(imgLen * 2, sizeof(float));
//   float *vCH2 = (float *) (vCH1 + imgLen);
//
//   float *vHatCH1 = (float *) calloc(imgLen * 2, sizeof(float));
//   float *vHatCH2 = (float *) (vHatCH1 + imgLen);
//
//   float *gradXvHatCH1 = (float *) calloc(imgLen * 4, sizeof(float));
//   float *gradYvHatCH1 = (float *) (gradXvHatCH1 + imgLen);
//   float *gradXvHatCH2 = (float *) (gradXvHatCH1 + imgLen * 2);
//   float *gradYvHatCH2 = (float *) (gradXvHatCH1 + imgLen * 3);
//
//   float *gradXuHat = (float *) calloc(imgLen * 2, sizeof(float));
//   float *gradYuHat = (float *) (gradXuHat + imgLen);
//
//   float theta = 1;
//
//   float *uTmp = (float *) malloc(sizeof(float) * imgLen);
//
//   float *vTmpCH1 = (float *) malloc(sizeof(float) * imgLen * 2);
//   float *vTmpCH2 = (float *) (vTmpCH1 + imgLen);
//
//   // Create new matcher simply to handle winner take all on updated matching errors
//   StereoMatcher *costs = allocMatcher(
//     self->left,
//     self->right,
//     self->disparityRange,
//     NULL, NULL, NULL, NULL, NULL
//   );
//
//   // Perform gradient descent
//   while(count < ITTERATION_LIMIT) {
//     printf("- TV itteration %d/%d\n", count, ITTERATION_LIMIT);
//
//     float aMax = 0;
//
//     for (int i = 0; i < imgLen; i++) {
//       // grad uHat
//       finiteForwardCH(gradXuHat + i, gradYuHat + i, uHat, width, height, i);
//
//
//
//       gCH1[i] = exp(-a[i] * gradBard[i]) * 2;
//       gCH4[i] = exp(-a[i] * gradBard[i]) * 2;
//
//       // grad vHat
//       finiteForwardCH(gradXvHatCH1 + i, gradYvHatCH1 + i, vHatCH1, width, height, i);
//       finiteForwardCH(gradXvHatCH2 + i, gradYvHatCH2 + i, vHatCH2, width, height, i);
//
//       // p^(n+1)
//       pCH1[i] = pCH1[i] + TAU_P * gCH1[i] * (gradXuHat[i] - vHatCH1[i]) +
//         TAU_P * gCH2[i] * (gradYuHat[i] - vHatCH2[i]);
//       pCH2[i] = pCH2[i] + TAU_P * gCH3[i] * (gradXuHat[i] - vHatCH1[i]) +
//         TAU_P * gCH4[i] * (gradYuHat[i] - vHatCH2[i]);
//
//       // q^(n+1)
//       qCH1[i] = qCH1[i] + TAU_Q * gradXvHatCH1[i];
//       qCH2[i] = qCH2[i] + TAU_Q * gradYvHatCH1[i];
//       qCH3[i] = qCH3[i] + TAU_Q * gradXvHatCH2[i];
//       qCH4[i] = qCH4[i] + TAU_Q * gradYvHatCH2[i];
//
//       float pNorm = maxD(fabs(pCH1[i]), fabs(pCH2[i]));
//       float pScale = 1.0f / maxD(1.0f, pNorm / LAMBDA_S);
//       pCH1[i] *= pScale;
//       pCH2[i] *= pScale;
//
//       float qNorm = maxD(fabs(qCH1[i]), maxD(fabs(qCH2[i]), maxD(fabs(qCH3[i]), fabs(qCH4[i]))));
//       float qScale = 1.0f / maxD(1.0f, qNorm / LAMBDA_A);
//       qCH1[i] *= qScale;
//       qCH2[i] *= qScale;
//       qCH3[i] *= qScale;
//       qCH4[i] *= qScale;
//     }
//
//     float *GxpCH1 = (float *) calloc(imgLen * 2, sizeof(float));
//     float *GxpCH2 = (float *) (GxpCH1 + imgLen);
//
//     for (int i = 0; i < imgLen; i++) {
//       // G=exp(-a * |grad D|) * Identity_2
//
//       float augmentedStepSize = TAU_U / theta;
//
//       GxpCH1[i] = gCH1[i] * pCH1[i] + gCH2[i] * pCH2[i];
//       GxpCH2[i] = gCH3[i] * pCH1[i] + gCH4[i] * pCH2[i];
//
//       float gradXGxpCH1, gradYGxpCH1,
//         gradXGxpCH2, gradYGxpCH2;
//
//       finiteBackwardCH(&gradXGxpCH1, &gradYGxpCH1, GxpCH1, width, i);
//       finiteBackwardCH(&gradXGxpCH2, &gradYGxpCH2, GxpCH2, width, i);
//
//       float divergenceGxp = gradXGxpCH1 + /*gradYGxpCH1 + gradXGxpCH2 +*/ gradYGxpCH2;
//
//       // u^(n+1)
//       uTmp[i] = u[i];
//       u[i] = (u[i] + TAU_U * divergenceGxp - TAU_U * L[i] + augmentedStepSize * a[i]) / (1 + augmentedStepSize);
//
//       u[i] = minD(u[i], 1.0f);
//       u[i] = maxD(u[i], 0.0f);
//
//       float gradXqCH1, gradYqCH1,
//         gradXqCH2, gradYqCH2,
//         gradXqCH3, gradYqCH3,
//         gradXqCH4, gradYqCH4;
//
//       finiteBackwardCH(&gradXqCH1, &gradYqCH1, qCH1, width, i);
//       finiteBackwardCH(&gradXqCH2, &gradYqCH2, qCH2, width, i);
//       finiteBackwardCH(&gradXqCH3, &gradYqCH3, qCH3, width, i);
//       finiteBackwardCH(&gradXqCH4, &gradYqCH4, qCH4, width, i);
//
//       float xxDiff = gradXqCH1;
//       float xyDiff = gradXqCH2;
//       float yxDiff = gradYqCH3;
//       float yyDiff = gradYqCH4;
//
//       float qDivergenceCH1 = xxDiff + yxDiff;
//       float qDivergenceCH2 = xyDiff + yyDiff;
//
//       vTmpCH1[i] = vCH1[i];
//       vTmpCH2[i] = vCH2[i];
//
//       // v^(n+1)
//       vCH1[i] = vCH1[i] + TAU_V * (pCH1[i] + qDivergenceCH1);
//       vCH2[i] = vCH2[i] + TAU_V * (pCH2[i] + qDivergenceCH2);
//     }
//
//     for (int i = 0; i < imgLen; i++) {
//       // u[i] /= uMax;
//       // u[i] = minD()
//
//       // uHat^(n+1)
//       uHat[i] = 2 * u[i] - uTmp[i];
//
//       // vHat^(n+1)
//       vHatCH1[i] = 2 * vCH1[i] - vTmpCH1[i];
//       vHatCH2[i] = 2 * vCH2[i] - vTmpCH2[i];
//
//     }
//
//     // a^(n+1)
//     for (int y = 0; y < height; y++) {
//       for (int x = 0; x < width; x++) {
//         int index = y * width + x;
//
//         for (int d = 0; d < self->disparityRange; d++) {
//           float cost = -1;
//
//           if (x - d >= 0) {
//             float a = 1.0f * d / self->disparityRange;
//             float diff = u[index] - a;
//             cost = LAMBDA_D * COST_AT(x, y, d, self) + L[index] * diff + diff * diff / (2 * theta);
//           }
//
//           COST_AT(x, y, d, costs) = cost;
//         }
//       }
//     }
//
//     WTA(costs);
//
//     for (int i = 0; i < imgLen; i++) {
//       a[i] = 1.0f * costs->disparityImg->pixels[i] / self->disparityRange;
//       costs->disparityImg->pixels[i] *= 4;
//       L[i] = L[i] + 1.0f / (2 * theta) * (u[i] - a[i]);
//     }
//
//     theta = theta * (1 - BETA * count);
//
//
//     char stageName[20];
//     sprintf(stageName, "stage%04d.pgm", count);
//     saveImg(stageName, costs->disparityImg);
//
//     count += 1;
//   }
//
//   for (int i = 0; i < imgLen; i++) {
//     self->disparityImg->pixels[i] = a[i] * self->disparityRange;
//   }
//
//   printf("finished\n");
// }
