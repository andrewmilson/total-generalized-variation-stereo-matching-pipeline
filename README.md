# Stereo implementation of total generalized variation (TGV) on GPU

This is a CUDA C implementation of [Fast and Accurate Large-Scale Stereo Reconstruction Using Variational Methods [2]](https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W21/html/Kuschk_Fast_and_Accurate_2013_ICCV_paper.html) and is supplied as part of the supplementary material to my honours thesis. Results below are from the first scene in the [2015 KITTI stere evaluate data set](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo).

| ![Cones animation](cones.webp) |
|:--:|
| *Animation of disparity images from each iteration of TGV's gradient descent* |

| ![KITTI animation](kitti.webp) |
|:--:|
| *Animation of disparity images from each iteration of TGV's gradient descent* |

<table>
  <tr><td>Left image</td><td><img src="kitti/reference.png" /></td></tr>
  <tr><td>Right image</td><td><img src="kitti/target.png" /></td></tr>
  <tr><td>Estimated disparity image<br /> <strong>MC</strong></td><td><img src="kitti/disparity-estimate-MC.png" /></td></tr>
  <tr><td>Estimated disparity image<br /> <strong>MC+AC</strong></td><td><img src="kitti/disparity-estimate-MC-AC.png" /></td></tr>
  <tr><td>Estimated disparity image<br /> <strong>MC+AC+TGV</strong></td><td><img src="kitti/disparity-estimate-MC-AC-TGV.png" /></td></tr>
  <tr><td>Ground truth disparity image</td><td><img src="kitti/ground-truth.png" /></td></tr>
</table>

+ Matching costs (MC)
+ Cost aggregation (AC[2])
+ Total generalized variation (TGV[1])

# Requirements

+ CUDA
+ CMake
+ Optional: ImageMagic to convert `.pgm` to `.png` -> `mogrify -format png *.pgm`

# Usage

```bash
# Build
mkdir build
cd build
cmake ../
make

# Binary descriptor representation of census transform on a 7x7 window
export CENSUS_DESCRIPTOR="90,144,91,144,92,144,93,144,94,144,95,144,\
96,144,107,144,108,144,109,144,110,144,111,144,112,144,113,144,124,144,\
125,144,126,144,127,144,128,144,129,144,130,144,141,144,142,144,143,144,\
144,144,145,144,146,144,147,144,158,144,159,144,160,144,161,144,162,144,\
163,144,164,144,175,144,176,144,177,144,178,144,179,144,180,144,181,144,\
192,144,193,144,194,144,195,144,196,144,197,144,198,144,144,144,144,144,\
144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,144,\
144,144,144,144,144,144,144,144"

# Usage:
# ./matching <left image> <right image> \
#   <estimated disparity image> <encoded binary descriptor> \
#   <max disparity> <disparity multiplyer>

# Pipeline:
# 1. Matching costs
# 2. Cost Aggregation[2]
# 3. TGV[1]
./matching ../kitti/reference.pgm ../kitti/target.pgm ../out.pgm \
  $CENSUS_DESCRIPTOR 128 1
```

# Limitations

+ The upper bound for the max disparity is 256. This can be increased by modifying the shared memory buffers of size 256 in [selection-functions.cu](src/selection-functions.cu).
+ Only greyscale `.pgm` format files can be used for input and output.
+ TGV parameter sets are specified at compile time. They are defined in [selection-functions.cu](src/selection-functions.cu).
```c++
...
// [1] mentions for low resolution Middleburry use λd=1.0 and λs=0.2
#define LAMBDA_S 0.2f
#define LAMBDA_A (8.0f * LAMBDA_S)
#define LAMBDA_D 1.0f
...
// [1] mentions for 2015 kitti use λd=0.4 and λs=1.0
#define LAMBDA_S 1.0f
#define LAMBDA_A (8.0f * LAMBDA_S)
#define LAMBDA_D 0.4f
...
```

# Related Publications

+ [1] Georg Kuschk and Daniel Cremers. 2013. [_"Fast and accurate large-scale stereo reconstruction using variational methods"_](https://www.cv-foundation.org/openaccess/content_iccv_workshops_2013/W21/html/Kuschk_Fast_and_Accurate_2013_ICCV_paper.html). In: Proceedings of the IEEE International Conference on Computer Vision Workshops pp. 700-707.
+ [2] Kuk-Jin Yoon and In So Kweon. 2006. [_"Adaptive support-weight approach forcorrespondence searc"_](). In: IEEE Transactions on Pattern Analysis and Machine Intelligence 28.4 (2006), pp. 650-656.

# What to cite

The paper related to this work has not been published yet. If you use this code in your research please cite:

```
@misc{milson2018,
  author = {Andrew Milson},
  title = {GPU Total Generalized Variation Stereo},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/andrewmilson/total-generalized-variation-stereo-matching-pipeline}},
}
```
