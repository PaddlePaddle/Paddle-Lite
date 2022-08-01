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

#include <iostream>
#include <vector>
#include "driver/nvidia_tensorrt/converter/plugin/yolo_box.h"
namespace nnadapter {
namespace nvidia_tensorrt {

YoloBoxPluginDynamic::YoloBoxPluginDynamic(const std::vector<int32_t>& anchors,
                                           int class_num,
                                           float conf_thresh,
                                           int downsample_ratio,
                                           bool clip_bbox,
                                           float scale_x_y,
                                           bool iou_aware,
                                           float iou_aware_factor)
    : anchors_(anchors),
      class_num_(class_num),
      conf_thresh_(conf_thresh),
      downsample_ratio_(downsample_ratio),
      clip_bbox_(clip_bbox),
      scale_x_y_(scale_x_y),
      iou_aware_(iou_aware),
      iou_aware_factor_(iou_aware_factor) {}

YoloBoxPluginDynamic::YoloBoxPluginDynamic(const void* serial_data,
                                           size_t serial_length) {
  Deserialize(&serial_data, &serial_length, &anchors_);
  Deserialize(&serial_data, &serial_length, &class_num_);
  Deserialize(&serial_data, &serial_length, &conf_thresh_);
  Deserialize(&serial_data, &serial_length, &downsample_ratio_);
  Deserialize(&serial_data, &serial_length, &clip_bbox_);
  Deserialize(&serial_data, &serial_length, &scale_x_y_);
  Deserialize(&serial_data, &serial_length, &iou_aware_);
  Deserialize(&serial_data, &serial_length, &iou_aware_factor_);
}

nvinfer1::IPluginV2DynamicExt* YoloBoxPluginDynamic::clone() const
    TRT_NOEXCEPT {
  return new YoloBoxPluginDynamic(anchors_,
                                  class_num_,
                                  conf_thresh_,
                                  downsample_ratio_,
                                  clip_bbox_,
                                  scale_x_y_,
                                  iou_aware_,
                                  iou_aware_factor_);
}

nvinfer1::DimsExprs YoloBoxPluginDynamic::getOutputDimensions(
    int32_t output_index,
    const nvinfer1::DimsExprs* inputs,
    int32_t nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  NNADAPTER_CHECK(inputs);
  nvinfer1::DimsExprs outdims;
  outdims.nbDims = 3;
  outdims.d[0] = inputs[0].d[0];
  int h = inputs[0].d[2]->getConstantValue();
  int w = inputs[0].d[3]->getConstantValue();
  int boxnum = h * w * anchors_.size() / 2;
  outdims.d[1] = expr_builder.constant(boxnum);
  if (output_index == 0) {
    outdims.d[2] = expr_builder.constant(4);
  } else if (output_index == 1) {
    outdims.d[2] = expr_builder.constant(class_num_);
  }
  return outdims;
}

template <typename T>
inline __device__ T Sigmoid(T x) {
  return (T)1. / ((T)1. + std::exp(-x));
}

template <typename T>
inline __device__ void GetYoloBox(T* box,
                                  const T* x,
                                  const int* anchors,
                                  int i,
                                  int j,
                                  int an_idx,
                                  int grid_size,
                                  int input_size,
                                  int index,
                                  int stride,
                                  int img_height,
                                  int img_width,
                                  T scale,
                                  T bias) {
  box[0] = (i + Sigmoid(x[index]) * scale + bias) * img_width / grid_size;
  box[1] =
      (j + Sigmoid(x[index + stride]) * scale + bias) * img_height / grid_size;
  box[2] = expf(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size;
  box[3] = expf(x[index + 3 * stride]) * anchors[2 * an_idx + 1] * img_height /
           input_size;
}

inline __device__ int GetEntryIndex(int batch,
                                    int an_idx,
                                    int hw_idx,
                                    int an_num,
                                    int an_stride,
                                    int stride,
                                    int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

template <typename T>
inline __device__ void CalcDetectionBox(T* boxes,
                                        T* box,
                                        const int box_idx,
                                        const int img_height,
                                        const int img_width,
                                        bool clip_bbox) {
  boxes[box_idx] = box[0] - box[2] / 2.f;
  boxes[box_idx + 1] = box[1] - box[3] / 2.f;
  boxes[box_idx + 2] = box[0] + box[2] / 2.f;
  boxes[box_idx + 3] = box[1] + box[3] / 2.f;

  if (clip_bbox) {
    boxes[box_idx] = boxes[box_idx] > 0 ? boxes[box_idx] : static_cast<T>(0);
    boxes[box_idx + 1] =
        boxes[box_idx + 1] > 0 ? boxes[box_idx + 1] : static_cast<T>(0);
    boxes[box_idx + 2] = boxes[box_idx + 2] < img_width - 1
                             ? boxes[box_idx + 2]
                             : static_cast<T>(img_width - 1);
    boxes[box_idx + 3] = boxes[box_idx + 3] < img_height - 1
                             ? boxes[box_idx + 3]
                             : static_cast<T>(img_height - 1);
  }
}

template <typename T>
inline __device__ void CalcLabelScore(T* scores,
                                      const T* input,
                                      const int label_idx,
                                      const int score_idx,
                                      const int class_num,
                                      const T conf,
                                      const int stride) {
  for (int i = 0; i < class_num; i++) {
    scores[score_idx + i] = conf * Sigmoid(input[label_idx + i * stride]);
  }
}

template <typename T, unsigned TPB>
__global__ void yolobox_kernel_value(int n,
                                     int h,
                                     int w,
                                     const float* input_data,
                                     const int* imgsize_data,
                                     float* boxes_data,
                                     float* scores_data,
                                     const int* anchors_data,
                                     int anchor_size,
                                     int class_num,
                                     float conf_thresh,
                                     int downsample_ratio,
                                     bool clip_bbox,
                                     float scale_x_y,
                                     bool iou_aware,
                                     float iou_aware_factor) {
  int idx = blockIdx.x * TPB + threadIdx.x;
  T bias = static_cast<T>(-0.5 * (scale_x_y - 1.));
  const int b_num = anchor_size / 2 * h * w;
  const int an_num = anchor_size / 2;
  int X_size = downsample_ratio * h;
  const int stride = h * w;
  const int an_stride = (class_num + 5) * stride;

  if (idx < n * b_num) {
    memset(&boxes_data[idx * 4], 0, 4 * sizeof(T));
    memset(&scores_data[idx * class_num], 0, class_num * sizeof(T));
    T box[4];
    int i = idx / b_num;                    // batch id
    int j = (idx % b_num) / (h * w);        // anchor id
    int k = ((idx % b_num) % (h * w)) / w;  // h id
    int l = ((idx % b_num) % (h * w)) % w;  // w id
    int img_height = imgsize_data[2 * i];
    int img_width = imgsize_data[2 * i + 1];
    int obj_idx = GetEntryIndex(i, j, k * w + l, an_num, an_stride, stride, 4);
    T conf = Sigmoid(input_data[obj_idx]);
    if (conf < conf_thresh) {
      return;
    }
    int box_idx = GetEntryIndex(i, j, k * w + l, an_num, an_stride, stride, 0);
    GetYoloBox(box,
               input_data,
               anchors_data,
               l,
               k,
               j,
               h,
               X_size,
               box_idx,
               stride,
               img_height,
               img_width,
               scale_x_y,
               bias);
    box_idx = (i * b_num + j * stride + k * w + l) * 4;
    CalcDetectionBox(
        boxes_data, box, box_idx, img_height, img_width, clip_bbox);

    int label_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, stride, 5);
    int score_idx = (i * b_num + j * stride + k * w + l) * class_num;
    CalcLabelScore(
        scores_data, input_data, label_idx, score_idx, class_num, conf, stride);
  }
}

int32_t YoloBoxPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  const int n = input_desc[0].dims.d[0];
  const int h = input_desc[0].dims.d[2];
  const int w = input_desc[0].dims.d[3];
  const int b_num = output_desc[0].dims.d[1];
  const int block_size = 256;
  const int grid_size = (n * b_num + block_size - 1) / block_size;
  const float* input_data = static_cast<const float*>(inputs[0]);
  const int* imgsize_data = static_cast<const int*>(inputs[1]);
  float* boxes_data = static_cast<float*>(outputs[0]);
  float* scores_data = static_cast<float*>(outputs[1]);

  int* dev_anchor_data;
  cudaMalloc(reinterpret_cast<void**>(&dev_anchor_data),
             anchors_.size() * sizeof(int));
  cudaMemcpy(dev_anchor_data,
             anchors_.data(),
             anchors_.size() * sizeof(int),
             cudaMemcpyHostToDevice);

  yolobox_kernel_value<float, block_size><<<grid_size, block_size, 0, stream>>>(
      n,
      h,
      w,
      input_data,
      imgsize_data,
      boxes_data,
      scores_data,
      dev_anchor_data,
      anchors_.size(),
      class_num_,
      conf_thresh_,
      downsample_ratio_,
      clip_bbox_,
      scale_x_y_,
      iou_aware_,
      iou_aware_factor_);
  cudaFree(dev_anchor_data);
  return 0;
}

size_t YoloBoxPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(anchors_) + sizeof(class_num_) + sizeof(conf_thresh_) +
         sizeof(downsample_ratio_) + sizeof(clip_bbox_) + sizeof(scale_x_y_) +
         sizeof(iou_aware_) + sizeof(iou_aware_factor_);
}

void YoloBoxPluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT {
  Serialize(&buffer, anchors_);
  Serialize(&buffer, class_num_);
  Serialize(&buffer, conf_thresh_);
  Serialize(&buffer, downsample_ratio_);
  Serialize(&buffer, clip_bbox_);
  Serialize(&buffer, scale_x_y_);
  Serialize(&buffer, iou_aware_);
  Serialize(&buffer, iou_aware_factor_);
}

int32_t YoloBoxPluginDynamic::getNbOutputs() const TRT_NOEXCEPT { return 2; }

nvinfer1::DataType YoloBoxPluginDynamic::getOutputDataType(
    int32_t index,
    const nvinfer1::DataType* input_types,
    int32_t nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(YoloBoxPluginDynamic,
                                   YoloBoxPluginDynamicCreator,
                                   "yolo_box_plugin_dynamic");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
