cmake_minimum_required(VERSION 2.8)
project(StereoMatching)

find_package(CUDA QUIET REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -use_fast_math
    -gencode=arch=compute_30,code=sm_30
    -gencode=arch=compute_35,code=sm_35
    -gencode=arch=compute_50,code=sm_50
    -gencode=arch=compute_52,code=sm_52
    -gencode=arch=compute_52,code=compute_52)

set(SRCS src/main.cu
  src/smooth-functions.cu
  src/match-functions.cu
  src/aggregation-functions.cu
  src/selection-functions.cu
  src/logger.cu
  src/stereo-matching.cu
  src/imageio.cu
  src/refine-functions.cu)

# add the executable
# add_executable(matching ${SRCS})
cuda_add_executable(matching ${SRCS})
set_target_properties(matching PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")
target_link_libraries(matching m)
