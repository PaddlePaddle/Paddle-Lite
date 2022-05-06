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

#include "driver/nvidia_tensorrt/kernel/cuda/yolo_box_post.h"
#include <algorithm>
#include <cmath>
#include "driver/nvidia_tensorrt/kernel/cuda/yolo_box_util.h"

namespace nnadapter {
namespace nvidia_tensorrt {
namespace cuda {

int YoloBoxPost::Run(
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
  auto anchors_count0 = anchors_operand0->length / (int)(sizeof(int32_t));
  auto anchors_count1 = anchors_operand1->length / (int)(sizeof(int32_t));
  auto anchors_count2 = anchors_operand2->length / (int)(sizeof(int32_t));
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
  cudaMalloc((void**)&d_anchors, anchors.size() * sizeof(int));
  cudaMemcpy(d_anchors,
             anchors.data(),
             anchors.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  int* dev_anchors_ptr[3];
  dev_anchors_ptr[0] = d_anchors;
  dev_anchors_ptr[1] = dev_anchors_ptr[0] + anchors_count0;
  dev_anchors_ptr[2] = dev_anchors_ptr[1] + anchors_count1;
  uint anchors_num[3] = {
      anchors_count0 / 2, anchors_count1 / 2, anchors_count2 / 2};
  /* various attrs */
  int class_num = *reinterpret_cast<int*>(input_operands[8]->buffer);
  float conf_thresh = *reinterpret_cast<float*>(input_operands[9]->buffer);
  int downsample_ratio0 = *reinterpret_cast<int*>(input_operands[10]->buffer);
  int downsample_ratio1 = *reinterpret_cast<int*>(input_operands[11]->buffer);
  int downsample_ratio2 = *reinterpret_cast<int*>(input_operands[12]->buffer);
  int downsample_ratio[3] = {
      downsample_ratio0, downsample_ratio1, downsample_ratio2};
  // clip_bbox[13] and scale_x_y[14] is not used now!
  // attrs with NMS
  float nms_thresh = *reinterpret_cast<float*>(input_operands[15]->buffer);

  // other attrs
  int batch = image_shape_tensor->Dims()[0];

  TensorInfo* ts_info = new TensorInfo[batch * boxes_input.size()];

  for (int i = 0; i < batch * boxes_input.size(); i++) {
    cudaMalloc(
        (void**)&ts_info[i].bboxes_dev_ptr,
        ts_info[i].bbox_count_max_alloc * (5 + class_num) * sizeof(float));
    ts_info[i].bboxes_host_ptr = (float*)malloc(
        ts_info[i].bbox_count_max_alloc * (5 + class_num) * sizeof(float));
    cudaMalloc((void**)&ts_info[i].bbox_count_device_ptr, sizeof(int));
  }

  // box index counter in gpu memory
  // *bbox_index_device_ptr used by atomicAdd
  int* bbox_index_device_ptr;
  cudaMalloc((void**)&bbox_index_device_ptr, sizeof(int));

  int total_bbox = 0;

  for (int batch_id = 0; batch_id < batch; batch_id++) {
    for (int input_id = 0; input_id < boxes_input.size(); input_id++) {
      int c = boxes_input_dims[input_id][1];
      int h = boxes_input_dims[input_id][2];
      int w = boxes_input_dims[input_id][3];
      int ts_id = batch_id * boxes_input.size() + input_id;
      int bbox_count_max_alloc = ts_info[ts_id].bbox_count_max_alloc;

      YoloTensorParseCuda(
          boxes_input[input_id] + batch_id * c * h * w,
          image_shape_data + batch_id * 2,
          image_scale_data + batch_id * 2,
          &(ts_info[ts_id].bboxes_dev_ptr),  // output in gpu,must use
          // 2-level pointer, because we may re-malloc
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
      cudaMemcpyAsync(
          ts_info[ts_id].bboxes_host_ptr,
          ts_info[ts_id].bboxes_dev_ptr,
          ts_info[ts_id].bbox_count_host * (5 + class_num) * sizeof(float),
          cudaMemcpyDeviceToHost);
      total_bbox += ts_info[ts_id].bbox_count_host;
    }
  }

  boxes_scores_tensor->Resize({total_bbox > 0 ? total_bbox : 1, 6});
  float* boxes_scores_data =
      reinterpret_cast<float*>(boxes_scores_tensor->Data(false));
  memset(boxes_scores_data, 0, sizeof(float) * 6);
  boxes_num_tensor->Resize({batch});
  int* boxes_num_data = reinterpret_cast<int*>(boxes_num_tensor->Data(false));
  int boxes_scores_id = 0;

  // NMS
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
        bbox_det.bbox.w =
            bbox_host_ptr[bbox_index * (5 + class_num) + 3] - bbox_det.bbox.x;
        bbox_det.bbox.h =
            bbox_host_ptr[bbox_index * (5 + class_num) + 4] - bbox_det.bbox.y;
        bbox_det.classes = class_num;
        bbox_det.prob = (float*)malloc(class_num * sizeof(float));
        int max_prob_class_id = -1;
        float max_class_prob = 0.0;
        for (int class_id = 0; class_id < class_num; class_id++) {
          float prob =
              bbox_host_ptr[bbox_index * (5 + class_num) + 5 + class_id];
          bbox_det.prob[class_id] = prob;
          if (prob > max_class_prob) {
            max_class_prob = prob;
            max_prob_class_id = class_id;
          }
        }
        bbox_det.max_prob_class_index = max_prob_class_id;
        bbox_det.sort_class = max_prob_class_id;
        bbox_det_vec.push_back(bbox_det);
      }
    }
    post_nms(bbox_det_vec, nms_thresh, class_num);
    for (int i = 0; i < bbox_det_vec.size(); i++) {
      boxes_scores_data[boxes_scores_id++] =
          bbox_det_vec[i].max_prob_class_index;
      boxes_scores_data[boxes_scores_id++] = bbox_det_vec[i].objectness;
      boxes_scores_data[boxes_scores_id++] = bbox_det_vec[i].bbox.x;
      boxes_scores_data[boxes_scores_id++] = bbox_det_vec[i].bbox.y;
      boxes_scores_data[boxes_scores_id++] =
          bbox_det_vec[i].bbox.w + bbox_det_vec[i].bbox.x;
      boxes_scores_data[boxes_scores_id++] =
          bbox_det_vec[i].bbox.h + bbox_det_vec[i].bbox.y;
      free(bbox_det_vec[i].prob);
    }
    boxes_num_data[batch_id] = bbox_det_vec.size();
  }

  cudaFree(bbox_index_device_ptr);
  for (int i = 0; i < batch * boxes_input.size(); i++) {
    cudaFree(ts_info[i].bboxes_dev_ptr);
    cudaFree(ts_info[i].bbox_count_device_ptr);
    free(ts_info[i].bboxes_host_ptr);
  }
  delete[] ts_info;
  return NNADAPTER_NO_ERROR;
}

}  // namespace cuda
}  // namespace nvidia_tensorrt
}  // namespace nnadapter
