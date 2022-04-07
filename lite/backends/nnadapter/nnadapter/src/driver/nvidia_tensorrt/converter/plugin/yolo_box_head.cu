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

__global__ void yolohead_kernel(const float* input,
                                float* output,
                                const uint gridSizeX,
                                const uint gridSizeY,
                                const uint numOutputClasses,
                                const uint numBBoxes,
                                const float scale_x_y) {
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
  uint z_id = blockIdx.z * blockDim.z + threadIdx.z;
  if ((x_id >= gridSizeX) || (y_id >= gridSizeY) || (z_id >= numBBoxes)) {
    return;
  }
  const int numGridCells = gridSizeX * gridSizeY;
  const int bbindex = y_id * gridSizeX + x_id;
  const float alpha = scale_x_y;
  const float beta = -0.5 * (scale_x_y - 1);

  output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)] =
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)] *
          alpha +
      beta;
  output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)] =
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)] *
          alpha +
      beta;
  output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)] = pow(
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)] * 2,
      2);
  output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)] = pow(
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)] * 2,
      2);
  output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)] =
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)];

  for (uint i = 0; i < numOutputClasses; ++i) {
    output[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))] =
        input[bbindex +
              numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))];
  }
}

cudaError_t YoloHead(const float* input,
                     float* output,
                     const uint gridSizeX,
                     const uint gridSizeY,
                     const uint numOutputClasses,
                     const uint numBBoxes,
                     const float scale_x_y,
                     cudaStream_t stream) {
  dim3 block(16, 16, 4);
  dim3 grid((gridSizeX / block.x) + 1,
            (gridSizeY / block.y) + 1,
            (numBBoxes / block.z) + 1);

  yolohead_kernel<<<grid, block, 0, stream>>>(input,
                                              output,
                                              gridSizeX,
                                              gridSizeY,
                                              numOutputClasses,
                                              numBBoxes,
                                              scale_x_y);

  return cudaGetLastError();
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
