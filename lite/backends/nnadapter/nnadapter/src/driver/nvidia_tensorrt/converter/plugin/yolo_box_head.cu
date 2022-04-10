// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "driver/nvidia_tensorrt/converter/plugin/yolo_box_head.h"

namespace nnadapter {
namespace nvidia_tensorrt {

inline __device__ float SigmoidGPU(const float& x) {
  return 1.0f / (1.0f + __expf(-x));
}

__global__ void YoloBoxHeadKernel(const float* input,
                                  float* output,
                                  const uint grid_size_x,
                                  const uint grid_size_y,
                                  const uint class_num,
                                  const uint anchors_num,
                                  const float scale_x_y) {
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
  uint z_id = blockIdx.z * blockDim.z + threadIdx.z;
  if ((x_id >= grid_size_x) || (y_id >= grid_size_y) || (z_id >= anchors_num)) {
    return;
  }
  const int grids_num = grid_size_x * grid_size_y;
  const int bbindex = y_id * grid_size_x + x_id;
  const float alpha = scale_x_y;
  const float beta = -0.5 * (scale_x_y - 1);

  output[bbindex + grids_num * (z_id * (5 + class_num) + 0)] =
      input[bbindex + grids_num * (z_id * (5 + class_num) + 0)] * alpha + beta;
  output[bbindex + grids_num * (z_id * (5 + class_num) + 1)] =
      input[bbindex + grids_num * (z_id * (5 + class_num) + 1)] * alpha + beta;
  output[bbindex + grids_num * (z_id * (5 + class_num) + 2)] =
      pow(input[bbindex + grids_num * (z_id * (5 + class_num) + 2)] * 2, 2);
  output[bbindex + grids_num * (z_id * (5 + class_num) + 3)] =
      pow(input[bbindex + grids_num * (z_id * (5 + class_num) + 3)] * 2, 2);
  output[bbindex + grids_num * (z_id * (5 + class_num) + 4)] =
      input[bbindex + grids_num * (z_id * (5 + class_num) + 4)];

  for (uint i = 0; i < class_num; ++i) {
    output[bbindex + grids_num * (z_id * (5 + class_num) + (5 + i))] =
        input[bbindex + grids_num * (z_id * (5 + class_num) + (5 + i))];
  }
}

__global__ void YoloBoxHeadV3Kernel(const float* input,
                                    float* output,
                                    const uint grid_size_x,
                                    const uint grid_size_y,
                                    const uint class_num,
                                    const uint anchors_num) {
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
  uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

  if ((x_id >= grid_size_x) || (y_id >= grid_size_y) || (z_id >= anchors_num)) {
    return;
  }

  const int grids_num = grid_size_x * grid_size_y;
  const int bbindex = y_id * grid_size_x + x_id;
  // x
  output[bbindex + grids_num * (z_id * (5 + class_num) + 0)] =
      SigmoidGPU(input[bbindex + grids_num * (z_id * (5 + class_num) + 0)]);
  // y
  output[bbindex + grids_num * (z_id * (5 + class_num) + 1)] =
      SigmoidGPU(input[bbindex + grids_num * (z_id * (5 + class_num) + 1)]);
  // w
  output[bbindex + grids_num * (z_id * (5 + class_num) + 2)] =
      __expf(input[bbindex + grids_num * (z_id * (5 + class_num) + 2)]);
  // h
  output[bbindex + grids_num * (z_id * (5 + class_num) + 3)] =
      __expf(input[bbindex + grids_num * (z_id * (5 + class_num) + 3)]);
  // objectness
  output[bbindex + grids_num * (z_id * (5 + class_num) + 4)] =
      SigmoidGPU(input[bbindex + grids_num * (z_id * (5 + class_num) + 4)]);

  // Probabilities of classes
  for (uint i = 0; i < class_num; ++i) {
    output[bbindex + grids_num * (z_id * (5 + class_num) + (5 + i))] =
        SigmoidGPU(
            input[bbindex + grids_num * (z_id * (5 + class_num) + (5 + i))]);
  }
}

cudaError_t YoloBoxHeadV3(const float* input,
                          float* output,
                          const int grid_size_x,
                          const int grid_size_y,
                          const int class_num,
                          const int anchors_num,
                          cudaStream_t stream) {
  dim3 block(16, 16, 4);
  dim3 grid((grid_size_x / block.x) + 1,
            (grid_size_y / block.y) + 1,
            (anchors_num / block.z) + 1);

  YoloBoxHeadV3Kernel<<<grid, block, 0, stream>>>(
      input, output, grid_size_x, grid_size_y, class_num, anchors_num);

  return cudaGetLastError();
}

cudaError_t YoloBoxHead(const float* input,
                        float* output,
                        const int grid_size_x,
                        const int grid_size_y,
                        const int class_num,
                        const int anchors_num,
                        const float scale_x_y,
                        cudaStream_t stream) {
  dim3 block(16, 16, 4);
  dim3 grid((grid_size_x / block.x) + 1,
            (grid_size_y / block.y) + 1,
            (anchors_num / block.z) + 1);

  YoloBoxHeadKernel<<<grid, block, 0, stream>>>(input,
                                                output,
                                                grid_size_x,
                                                grid_size_y,
                                                class_num,
                                                anchors_num,
                                                scale_x_y);

  return cudaGetLastError();
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
