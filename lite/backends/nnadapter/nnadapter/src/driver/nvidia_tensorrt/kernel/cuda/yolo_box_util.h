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

#pragma once
#include <cuda_runtime_api.h>
#include <map>
#include <memory>
#include <vector>

namespace nnadapter {
namespace nvidia_tensorrt {
namespace cuda {
typedef struct { float x, y, w, h; } box;

typedef struct detection {
  box bbox;
  int classes;
  float* prob;
  float* mask;
  float objectness;
  int sort_class;
  int max_prob_class_index;
} detection;

typedef struct TensorInfo {
  int bbox_count_host;  // record bbox numbers
  int bbox_count_max_alloc{50};
  float* bboxes_dev_ptr;
  float* bboxes_host_ptr;
  int* bbox_count_device_ptr;  // box counter in gpu memory, used by atomicAdd
} TensorInfo;

void post_nms(std::vector<detection>& det_bboxes, float thresh, int classes);

cudaError_t YoloTensorParseCuda(
    const float* input_data,  // [in] YOLO layer tensor input
    const float* image_shape_data,
    const float* image_scale_data,
    float** bboxes_tensor_ptr,  // [out] Bounding boxes output tensor
    int& bbox_count_max_alloc,  // [in/out] maximum bounding box number
                                // allocated in dev
    int& bbox_count_host,  // [in/out] bounding boxes number recorded in host
    int* bbox_count_device_ptr,  // [in/out] bounding boxes number calculated in
                                 // device side
    int* bbox_index_device_ptr,  // [in] bounding box index for kernel threads
                                 // shared access
    int grid_size,
    int class_num,
    int anchors_num,
    int netw,
    int neth,
    int* biases_device,
    float prob_thresh);

}  // namespace cuda
}  // namespace nvidia_tensorrt
}  // namespace nnadapter
