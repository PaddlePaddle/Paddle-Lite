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

#include "driver/nvidia_tensorrt/kernel/cuda/yolo_box_util.h"

namespace nnadapter {
namespace nvidia_tensorrt {
namespace cuda {

__global__ void YoloBoxNum(const float* input,
                           int* bbox_count,
                           const uint grid_size,
                           const uint class_num,
                           const uint anchors_num,
                           float prob_thresh) {
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
  uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

  if ((x_id >= grid_size) || (y_id >= grid_size) || (z_id >= anchors_num)) {
    return;
  }

  const int grids_num = grid_size * grid_size;
  const int bbindex = y_id * grid_size + x_id;

  // objectness
  float objectness = input[bbindex + grids_num * (z_id * (5 + class_num) + 4)];
  if (objectness < prob_thresh) {
    return;
  }

  atomicAdd(bbox_count, 1);
}

inline __device__ void CorrectYoloBox(float& x,
                                      float& y,
                                      float& w,
                                      float& h,
                                      float pic_w,
                                      float pic_h,
                                      float netw,
                                      float neth) {
  int new_w = 0;
  int new_h = 0;
  if ((netw / pic_w) < (neth / pic_h)) {
    new_w = netw;
    new_h = (pic_h * netw) / pic_w;
  } else {
    new_h = neth;
    new_w = (pic_w * neth) / pic_h;
  }

  x = (x - (netw - new_w) / 2.) / new_w;
  y = (y - (neth - new_h) / 2.) / new_h;
  w /= (float)new_w;
  h /= (float)new_h;
}

__global__ void YoloTensorParseKernel(const float* input,
                                      const float* im_shape_data,
                                      const float* im_scale_data,
                                      float* output,
                                      int* bbox_index,
                                      const uint grid_size,
                                      const uint class_num,
                                      const uint anchors_num,
                                      const uint netw,
                                      const uint neth,
                                      int* biases,
                                      float prob_thresh) {
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
  uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

  if ((x_id >= grid_size) || (y_id >= grid_size) || (z_id >= anchors_num)) {
    return;
  }

  const float pic_h = im_shape_data[0] / im_scale_data[0];
  const float pic_w = im_shape_data[1] / im_scale_data[1];

  const int grids_num = grid_size * grid_size;
  const int bbindex = y_id * grid_size + x_id;

  // objectness
  float objectness = input[bbindex + grids_num * (z_id * (5 + class_num) + 4)];

  if (objectness < prob_thresh) {
    return;
  }

  int cur_bbox_index = atomicAdd(bbox_index, 1);
  int tensor_index = cur_bbox_index * (5 + class_num);

  // x
  float x = input[bbindex + grids_num * (z_id * (5 + class_num) + 0)];
  x = (float)((x + (float)x_id) * (float)netw) / (float)grid_size;

  // y
  float y = input[bbindex + grids_num * (z_id * (5 + class_num) + 1)];
  y = (float)((y + (float)y_id) * (float)neth) / (float)grid_size;

  // w
  float w = input[bbindex + grids_num * (z_id * (5 + class_num) + 2)];
  w = w * biases[2 * z_id];

  // h
  float h = input[bbindex + grids_num * (z_id * (5 + class_num) + 3)];
  h = h * biases[2 * z_id + 1];

  CorrectYoloBox(x, y, w, h, pic_w, pic_h, netw, neth);

  output[tensor_index] = objectness;
  output[tensor_index + 1] = x;
  output[tensor_index + 2] = y;
  output[tensor_index + 3] = w;
  output[tensor_index + 4] = h;

  // Probabilities of classes
  for (uint i = 0; i < class_num; ++i) {
    float prob =
        input[bbindex + grids_num * (z_id * (5 + class_num) + (5 + i))] *
        objectness;
    output[tensor_index + 5 + i] = prob < prob_thresh ? 0. : prob;
  }
}

cudaError_t YoloTensorParseCuda(
    const float* input_data,  // [in] YOLO_BOX_HEAD layer output
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
    float prob_thresh) {
  dim3 threads_per_block(16, 16, 4);
  dim3 number_of_blocks((grid_size / threads_per_block.x) + 1,
                        (grid_size / threads_per_block.y) + 1,
                        (anchors_num / threads_per_block.z) + 1);

  // Estimate how many boxes will be choosed
  int bbox_count = 0;
  cudaMemcpy(
      bbox_count_device_ptr, &bbox_count, sizeof(int), cudaMemcpyHostToDevice);
  YoloBoxNum<<<number_of_blocks, threads_per_block, 0>>>(input_data,
                                                         bbox_count_device_ptr,
                                                         grid_size,
                                                         class_num,
                                                         anchors_num,
                                                         prob_thresh);
  cudaMemcpy(
      &bbox_count, bbox_count_device_ptr, sizeof(int), cudaMemcpyDeviceToHost);

  // Record actual bbox number
  bbox_count_host = bbox_count;

  // Obtain previous allocated bbox tensor in device side
  float* bbox_tensor = *bboxes_tensor_ptr;
  // Update previous maximum bbox number
  if (bbox_count > bbox_count_max_alloc) {
    printf(
        "Bbox tensor expanded: %d -> %d!\n", bbox_count_max_alloc, bbox_count);
    cudaFree(bbox_tensor);
    cudaMalloc(&bbox_tensor, bbox_count * (5 + class_num) * sizeof(float));
    bbox_count_max_alloc = bbox_count;
    *bboxes_tensor_ptr = bbox_tensor;
  }

  // Now generate bboxes
  int bbox_index = 0;
  cudaMemcpy(
      bbox_index_device_ptr, &bbox_index, sizeof(int), cudaMemcpyHostToDevice);
  YoloTensorParseKernel<<<number_of_blocks, threads_per_block, 0>>>(
      input_data,
      image_shape_data,
      image_scale_data,
      bbox_tensor,
      bbox_index_device_ptr,
      grid_size,
      class_num,
      anchors_num,
      netw,
      neth,
      biases_device,
      prob_thresh);

  cudaError_t status = cudaGetLastError();
  if (cudaSuccess != status) {
    printf("yolo_tensor_parse_cuda error: %s\n", cudaGetErrorString(status));
  }

  return status;
}
}  // namespace cuda
}  // namespace nvidia_tensorrt
}  // namespace nnadapter
