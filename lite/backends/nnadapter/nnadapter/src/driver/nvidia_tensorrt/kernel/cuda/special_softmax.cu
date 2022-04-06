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

#include <algorithm>
#include <cmath>
#include "driver/nvidia_tensorrt/kernel/cuda/special_softmax.h"
#include "driver/nvidia_tensorrt/kernel/cuda/yolo.h"
namespace nnadapter {
namespace nvidia_tensorrt {
namespace cuda {

__global__ void yolo_tensor_bbox_num(const float* input,
                                     int* bbox_count,
                                     const uint gridSize,
                                     const uint numOutputClasses,
                                     const uint numBBoxes,
                                     float prob_thresh) {
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
  uint z_id = blockIdx.z * blockDim.z + threadIdx.z;
  if (x_id == 0 && y_id == 0 && z_id == 0) printf("%f\n", *input);

  if ((x_id >= gridSize) || (y_id >= gridSize) || (z_id >= numBBoxes)) {
    return;
  }

  const int numGridCells = gridSize * gridSize;
  const int bbindex = y_id * gridSize + x_id;

  // objectness
  float objectness =
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)];
  if (objectness < prob_thresh) {
    return;
  }

  atomicAdd(bbox_count, 1);
}

inline __device__ void correct_yolo_box(float& x,
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

__global__ void yolo_tensor_parse_kernel(const float* input,
                                         const float* ImShape_data,
                                         const float* ScaleFactor_data,
                                         float* output,
                                         int* bbox_index,
                                         const uint gridSize,
                                         const uint numOutputClasses,
                                         const uint numBBoxes,
                                         const uint netw,
                                         const uint neth,
                                         int* biases,
                                         float prob_thresh) {
  uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
  uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

  if ((x_id >= gridSize) || (y_id >= gridSize) || (z_id >= numBBoxes)) {
    return;
  }

  const float pic_h = ImShape_data[0] / ScaleFactor_data[0];
  const float pic_w = ImShape_data[1] / ScaleFactor_data[1];

  const int numGridCells = gridSize * gridSize;
  const int bbindex = y_id * gridSize + x_id;

  // objectness
  float objectness =
      input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)];

  if (objectness < prob_thresh) {
    return;
  }

  int cur_bbox_index = atomicAdd(bbox_index, 1);
  int tensor_index = cur_bbox_index * (5 + numOutputClasses);

  // x
  float x = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)];
  x = (float)((x + (float)x_id) * (float)netw) / (float)gridSize;

  // y
  float y = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)];
  y = (float)((y + (float)y_id) * (float)neth) / (float)gridSize;

  // w
  float w = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)];
  w = w * biases[2 * z_id];

  // h
  float h = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)];
  h = h * biases[2 * z_id + 1];

  correct_yolo_box(x, y, w, h, pic_w, pic_h, netw, neth);

  output[tensor_index] = objectness;
  output[tensor_index + 1] = x;
  output[tensor_index + 2] = y;
  output[tensor_index + 3] = w;
  output[tensor_index + 4] = h;

  // Probabilities of classes
  for (uint i = 0; i < numOutputClasses; ++i) {
    float prob =
        input[bbindex +
              numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))] *
        objectness;
    output[tensor_index + 5 + i] = prob < prob_thresh ? 0. : prob;
  }
}

void yolo_tensor_parse_cuda(
    const float* yolo_layer_tensor,  // [in] YOLO layer tensor input
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
    int gridSize,
    int numOutputClasses,
    int numBBoxes,
    int netw,
    int neth,
    int* biases_device,
    float prob_thresh) {
  dim3 threads_per_block(16, 16, 4);
  dim3 number_of_blocks((gridSize / threads_per_block.x) + 1,
                        (gridSize / threads_per_block.y) + 1,
                        (numBBoxes / threads_per_block.z) + 1);

  // evaluated how many boxes will be choosed
  int bbox_count = 0;
  cudaMemcpy(
      bbox_count_device_ptr, &bbox_count, sizeof(int), cudaMemcpyHostToDevice);
  yolo_tensor_bbox_num<<<number_of_blocks, threads_per_block, 0>>>(
      yolo_layer_tensor,
      bbox_count_device_ptr,
      gridSize,
      numOutputClasses,
      numBBoxes,
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
    cudaMalloc(&bbox_tensor,
               bbox_count * (5 + numOutputClasses) * sizeof(float));
    bbox_count_max_alloc = bbox_count;
    *bboxes_tensor_ptr = bbox_tensor;
  }

  // now generate the boxes
  int bbox_index = 0;
  cudaMemcpy(
      bbox_index_device_ptr, &bbox_index, sizeof(int), cudaMemcpyHostToDevice);
  yolo_tensor_parse_kernel<<<number_of_blocks, threads_per_block, 0>>>(
      yolo_layer_tensor,
      image_shape_data,
      image_scale_data,
      bbox_tensor,
      bbox_index_device_ptr,
      gridSize,
      numOutputClasses,
      numBBoxes,
      netw,
      neth,
      biases_device,
      prob_thresh);
}

int SpecialSoftmaxKernel::Run(
    core::Operation* operation,
    std::map<core::Operand*, std::shared_ptr<Tensor>>* operand_map) {
  auto& input_operands = operation->input_operands;
  std::vector<const float*> boxes_input;
  std::vector<std::vector<int32_t>> boxes_input_dims;
  for (int i = 0; i < 3; i++) {
    auto input_tensor = operand_map->at(operation->input_operands[i]);
    const float* input = reinterpret_cast<const float*>(input_tensor->Data());
    boxes_input.push_back(input);
    boxes_input_dims.push_back(input_tensor->Dims());
  }
  auto image_shape_tensor = operand_map->at(input_operands[3]);
  auto image_scale_tensor = operand_map->at(input_operands[4]);
  const float* image_shape_data =
      reinterpret_cast<const float*>(image_shape_tensor->Data());
  const float* image_scale_data =
      reinterpret_cast<const float*>(image_scale_tensor->Data());

  auto boxes_scores_tensor = operand_map->at(operation->output_operands[0]);
  auto boxes_num_tensor = operand_map->at(operation->output_operands[1]);
  /* anchors */
  auto anchors_operand0 = input_operands[5];
  auto anchors_operand1 = input_operands[6];
  auto anchors_operand2 = input_operands[7];
  auto anchors_count0 = anchors_operand0->length / sizeof(int32_t);
  auto anchors_count1 = anchors_operand1->length / sizeof(int32_t);
  auto anchors_count2 = anchors_operand2->length / sizeof(int32_t);
  auto anchors_data0 = reinterpret_cast<int32_t*>(anchors_operand0->buffer);
  auto anchors_data1 = reinterpret_cast<int32_t*>(anchors_operand1->buffer);
  auto anchors_data2 = reinterpret_cast<int32_t*>(anchors_operand2->buffer);
  auto anchors =
      std::vector<int32_t>(anchors_count0 + anchors_count1 + anchors_count2);

  memcpy(&anchors[0], anchors_data0, anchors_count0 * sizeof(int));
  memcpy(&anchors[anchors_count0], anchors_data1, anchors_count1 * sizeof(int));
  memcpy(&anchors[anchors_count0 + anchors_count1],
         anchors_data2,
         anchors_count2 * sizeof(int));
  // memcpy anchors to gpu memory
  int* d_anchors;
  cudaMalloc(&d_anchors, anchors.size() * sizeof(int));
  cudaMemcpy(d_anchors,
             anchors.data(),
             anchors.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  int* dev_anchors_ptr[3];
  dev_anchors_ptr[0] = d_anchors;
  dev_anchors_ptr[1] = dev_anchors_ptr[0] + anchors_count0;
  dev_anchors_ptr[2] = dev_anchors_ptr[1] + anchors_count1;
  int anchors_num[3] = {
      anchors_count0 / 2, anchors_count1 / 2, anchors_count2 / 2};
  /* various attrs */
  int class_num = *reinterpret_cast<int*>(input_operands[8]->buffer);
  float conf_thresh = *reinterpret_cast<float*>(input_operands[9]->buffer);
  int downsample_ratio0 = *reinterpret_cast<int*>(input_operands[10]->buffer);
  int downsample_ratio1 = *reinterpret_cast<int*>(input_operands[11]->buffer);
  int downsample_ratio2 = *reinterpret_cast<int*>(input_operands[12]->buffer);
  int downsample_ratio[3] = {
      downsample_ratio0, downsample_ratio1, downsample_ratio2};
  // clip_bbox and scale_x_y is not used now!
  bool clip_bbox = *reinterpret_cast<bool*>(input_operands[13]->buffer);
  float scale_x_y = *reinterpret_cast<float*>(input_operands[14]->buffer);
  // attrs with NMS
  float nms_thresh = *reinterpret_cast<float*>(input_operands[15]->buffer);

  // other attrs
  int batch = image_shape_tensor->Dims()[0];

  TensorInfo* ts_info = new TensorInfo[batch * boxes_input.size()];

  for (int i = 0; i < batch * boxes_input.size(); i++) {
    cudaMalloc(
        &ts_info[i].bboxes_dev_ptr,
        ts_info[i].bbox_count_max_alloc * (5 + class_num) * sizeof(float));
    ts_info[i].bboxes_host_ptr =
        (float*)malloc(ts_info[i].bbox_count_max_alloc * sizeof(float));
    cudaMalloc(&ts_info[i].bbox_count_device_ptr, sizeof(int));
  }

  // box index counter in gpu memory
  // *bbox_index_device_ptr used by atomicAdd
  int* bbox_index_device_ptr;
  cudaMalloc(&bbox_index_device_ptr, sizeof(int));

  std::vector<float> result;

  for (int batch_id = 0; batch_id < batch; batch_id++) {
    for (int input_id = 0; input_id < boxes_input.size(); input_id++) {
      int c = boxes_input_dims[input_id][1];
      int h = boxes_input_dims[input_id][2];
      int w = boxes_input_dims[input_id][3];
      int ts_id = batch_id * boxes_input.size() + input_id;
      int bbox_count_max_alloc = ts_info[ts_id].bbox_count_max_alloc;
      yolo_tensor_parse_cuda(
          boxes_input[input_id] + batch_id * c * h * w,
          image_shape_data + batch_id * 2,
          image_scale_data + batch_id * 2,
          &(ts_info[ts_id].bboxes_dev_ptr),  // output in gpu memory,
                                             // here we must use
                                             // 2-level
          // pointer, because we may re-malloc this area
          bbox_count_max_alloc,  // bbox_count_alloc_ptr boxes we pre-allocate
          ts_info[ts_id].bbox_count_host,        // record bbox numbers
          ts_info[ts_id].bbox_count_device_ptr,  // for atomicAdd
          bbox_index_device_ptr,                 // for atomicAdd
          h,
          class_num,
          anchors_num[input_id],
          downsample_ratio[input_id] * h,
          downsample_ratio[input_id] * w,
          dev_anchors_ptr[input_id],
          conf_thresh);

      // batch info update
      if (bbox_count_max_alloc > ts_info[ts_id].bbox_count_max_alloc) {
        ts_info[ts_id].bbox_count_max_alloc = bbox_count_max_alloc;
        ts_info[ts_id].bboxes_host_ptr = (float*)realloc(
            ts_info[ts_id].bboxes_host_ptr,
            bbox_count_max_alloc * (5 + class_num) * sizeof(float));
      }
      // we need copy bbox_count_host boxes to cpu memory
      cudaMemcpy(
          ts_info[ts_id].bboxes_host_ptr,
          ts_info[ts_id].bboxes_dev_ptr,
          ts_info[ts_id].bbox_count_host * (5 + class_num) * sizeof(float),
          cudaMemcpyDeviceToHost);

      // copy bbox_count_host boxes to cpu memory
      for (int i = 0; i < ts_info[ts_id].bbox_count_host * (5 + class_num); i++)
        result.push_back(ts_info[ts_id].bboxes_host_ptr[i]);
    }
  }
  boxes_scores_tensor->Resize(
      {(int)(result.size() / (5 + class_num)), 5 + class_num});
  float* boxes_scores_data =
      reinterpret_cast<float*>(boxes_scores_tensor->Data(false));
  memcpy(boxes_scores_data, result.data(), result.size() * sizeof(float));

  boxes_num_tensor->Resize({batch});
  int* boxes_num_data = reinterpret_cast<int*>(boxes_num_tensor->Data(false));

  /* ----------------------- NMS  -----------------------------  */

  for (int batch_id = 0; batch_id < batch; batch_id++) {
    std::vector<detection> bbox_det_vec;

    for (int input_id = 0; input_id < boxes_input.size(); input_id++) {
      int ts_id = batch_id * boxes_input.size() + input_id;
      int bbox_count = ts_info[ts_id].bbox_count_host;

      if (bbox_count <= 0) {
        continue;
      }

      float* bbox_host_ptr = ts_info[ts_id].bboxes_host_ptr;

      for (int bbox_index = 0; bbox_index < bbox_count; ++bbox_index) {
        detection bbox_det;
        memset(&bbox_det, 0, sizeof(detection));
        bbox_det.objectness = bbox_host_ptr[bbox_index * (5 + class_num) + 0];
        bbox_det.bbox.x = bbox_host_ptr[bbox_index * (5 + class_num) + 1];
        bbox_det.bbox.y = bbox_host_ptr[bbox_index * (5 + class_num) + 2];
        bbox_det.bbox.w = bbox_host_ptr[bbox_index * (5 + class_num) + 3];
        bbox_det.bbox.h = bbox_host_ptr[bbox_index * (5 + class_num) + 4];
        bbox_det.classes = class_num;
        bbox_det.prob = (float*)malloc(class_num * sizeof(float));
        int max_prob_class_index = -1;
        float max_class_prob = 0.0;
        for (int class_index = 0; class_index < class_num; class_index++) {
          float prob =
              bbox_host_ptr[bbox_index * (5 + class_num) + 5 + class_index];
          bbox_det.prob[class_index] = prob;
          if (prob > max_class_prob) {
            max_class_prob = prob;
            max_prob_class_index = class_index;
          }
        }
        bbox_det.max_prob_class_index = max_prob_class_index;
        bbox_det.sort_class = max_prob_class_index;
        bbox_det_vec.push_back(bbox_det);
      }
    }
    post_nms(bbox_det_vec, nms_thresh, class_num);
    for (int i = 0; i < bbox_det_vec.size(); i++)
    free(bbox_det_vec[i].prob);
  
    boxes_num_data[batch_id] = 1;  // bbox_det_vec.size();
  }
  /* ----------------------- NMS END  -----------------------------  */

  /* ----------------------- for  -----------------------------  */
  struct Num {
    float data[85];
  };

  std::vector<Num> tmp_result;

  for (int i = 0; i < result.size(); i += (5 + class_num)) {
    Num ele;

    for (int j = 0; j < (5 + class_num); j++) {
      ele.data[j] = result[i + j];
    }
    tmp_result.push_back(ele);
  }

  std::sort(tmp_result.begin(), tmp_result.end(), [](Num a, Num b) {
    return a.data[0] > b.data[0];
  });
  result.clear();
  for (int i = 0; i < tmp_result.size(); i++)
    for (int j = 0; j < 85; j++) result.push_back(tmp_result[i].data[j]);

  memcpy(boxes_scores_data, result.data(), result.size() * sizeof(float));

  FILE* f = fopen("/zhoukangkang/lite_jetson_yolo_head/cuda.txt", "w");
  for (int i = 0; i < result.size(); i++) fprintf(f, "%f\n", result[i]);
  fclose(f);
  /*-------------------------------------------------------------------------*/

  cudaFree(bbox_index_device_ptr);

  for (int i = 0; i < batch * boxes_input.size(); i++) {
    cudaFree(ts_info[i].bboxes_dev_ptr);
    cudaFree(ts_info[i].bbox_count_device_ptr);
    free(ts_info[i].bboxes_host_ptr);
  }

  return NNADAPTER_NO_ERROR;
}

}  // namespace cuda
}  // namespace nvidia_tensorrt
}  // namespace nnadapter
