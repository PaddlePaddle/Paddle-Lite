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
#include <vector>
#include "driver/nvidia_tensorrt/converter/plugin/prior_box.h"

namespace nnadapter {
namespace nvidia_tensorrt {

PriorBoxPluginDynamic::PriorBoxPluginDynamic() {}

PriorBoxPluginDynamic::PriorBoxPluginDynamic(
    const std::vector<float>& aspect_ratios,
    const std::vector<int32_t>& input_dimension,
    const std::vector<int32_t>& image_dimension,
    float step_w,
    float step_h,
    const std::vector<float>& min_sizes,
    const std::vector<float>& max_sizes,
    float offset,
    bool is_clip,
    bool is_flip,
    bool min_max_aspect_ratios_order,
    const std::vector<float>& variances)
    : aspect_ratios_(aspect_ratios),
      input_dimension_(input_dimension),
      image_dimension_(image_dimension),
      step_w_(step_w),
      step_h_(step_h),
      min_sizes_(min_sizes),
      max_sizes_(max_sizes),
      offset_(offset),
      is_clip_(is_clip),
      is_flip_(is_flip),
      min_max_aspect_ratios_order_(min_max_aspect_ratios_order),
      variances_(variances) {}

PriorBoxPluginDynamic::PriorBoxPluginDynamic(const void* serial_data,
                                             size_t serial_length) {
  Deserialize(&serial_data, &serial_length, &aspect_ratios_);
  Deserialize(&serial_data, &serial_length, &input_dimension_);
  Deserialize(&serial_data, &serial_length, &image_dimension_);
  Deserialize(&serial_data, &serial_length, &step_w_);
  Deserialize(&serial_data, &serial_length, &step_h_);
  Deserialize(&serial_data, &serial_length, &min_sizes_);
  Deserialize(&serial_data, &serial_length, &max_sizes_);
  Deserialize(&serial_data, &serial_length, &offset_);
  Deserialize(&serial_data, &serial_length, &is_clip_);
  Deserialize(&serial_data, &serial_length, &is_flip_);
  Deserialize(&serial_data, &serial_length, &min_max_aspect_ratios_order_);
  Deserialize(&serial_data, &serial_length, &variances_);
}

nvinfer1::IPluginV2DynamicExt* PriorBoxPluginDynamic::clone() const
    TRT_NOEXCEPT {
  return new PriorBoxPluginDynamic(aspect_ratios_,
                                   input_dimension_,
                                   image_dimension_,
                                   step_w_,
                                   step_h_,
                                   min_sizes_,
                                   max_sizes_,
                                   offset_,
                                   is_clip_,
                                   is_flip_,
                                   min_max_aspect_ratios_order_,
                                   variances_);
}

template <typename T>
__device__ inline T clip(T in) {
  return min(max(in, 0.), 1.);
}

template <typename T, unsigned TPB>
__global__ void SetVariance(T* out, T* var, int vnum, int num) {
  int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < num) {
    out[idx] = var[idx % vnum];
  }
}

template <typename T, unsigned TPB>
__global__ void prior_box_kernel(T* out,
                                 const T* aspect_ratios,
                                 const int height,
                                 const int width,
                                 const int im_height,
                                 const int im_width,
                                 const std::size_t as_num,
                                 const T offset,
                                 const T step_width,
                                 const T step_height,
                                 const T* min_sizes,
                                 const T* max_sizes,
                                 const int max_num,
                                 const int min_num,
                                 bool is_clip,
                                 bool min_max_aspect_ratios_order) {
  int num_priors = max_sizes ? as_num * min_num + min_num : as_num * min_num;
  int box_num = height * width * num_priors;
  int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < box_num) {
    int h = idx / (num_priors * width);
    int w = (idx / num_priors) % width;
    int p = idx % num_priors;
    int m = max_sizes ? p / (as_num + 1) : p / as_num;
    T cx = (w + offset) * step_width;
    T cy = (h + offset) * step_height;
    T bw, bh;
    T min_size = min_sizes[m];
    if (max_num) {
      int s = p % (as_num + 1);
      if (!min_max_aspect_ratios_order) {
        if (s < as_num) {
          T ar = aspect_ratios[s];
          bw = min_size * sqrt(ar) / 2.;
          bh = min_size / sqrt(ar) / 2.;
        } else {
          T max_size = max_sizes[m];
          bw = sqrt(min_size * max_size) / 2.;
          bh = bw;
        }
      } else {
        if (s == 0) {
          bw = bh = min_size / 2.;
        } else if (s == 1) {
          T max_size = max_sizes[m];
          bw = sqrt(min_size * max_size) / 2.;
          bh = bw;
        } else {
          T ar = aspect_ratios[s - 1];
          bw = min_size * sqrt(ar) / 2.;
          bh = min_size / sqrt(ar) / 2.;
        }
      }
    } else {
      int s = p % as_num;
      T ar = aspect_ratios[s];
      bw = min_size * sqrt(ar) / 2.;
      bh = min_size / sqrt(ar) / 2.;
    }
    T xmin = (cx - bw) / im_width;
    T ymin = (cy - bh) / im_height;
    T xmax = (cx + bw) / im_width;
    T ymax = (cy + bh) / im_height;
    out[idx * 4] = is_clip ? clip<T>(xmin) : xmin;
    out[idx * 4 + 1] = is_clip ? clip<T>(ymin) : ymin;
    out[idx * 4 + 2] = is_clip ? clip<T>(xmax) : xmax;
    out[idx * 4 + 3] = is_clip ? clip<T>(ymax) : ymax;
  }
}
nvinfer1::DimsExprs PriorBoxPluginDynamic::getOutputDimensions(
    int32_t output_index,
    const nvinfer1::DimsExprs* inputs,
    int32_t nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  NNADAPTER_CHECK(inputs);
  NNADAPTER_CHECK_GE(nb_inputs, 2);
  nvinfer1::DimsExprs outdims;
  outdims.nbDims = 4;
  std::vector<float> aspect_ratios_vec;
  ExpandAspectRatios(aspect_ratios_, is_flip_, &aspect_ratios_vec);
  int num_priors = aspect_ratios_vec.size() * min_sizes_.size();
  if (max_sizes_.size() > 0) {
    num_priors += max_sizes_.size();
  }
  outdims.d[0] = expr_builder.constant(input_dimension_[2]);
  outdims.d[1] = expr_builder.constant(input_dimension_[3]);
  outdims.d[2] = expr_builder.constant(num_priors);
  outdims.d[3] = expr_builder.constant(4);
  return outdims;
}

int32_t PriorBoxPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto height = input_dimension_[2];
  auto width = input_dimension_[3];
  auto in_height = image_dimension_[2];
  auto in_width = image_dimension_[3];
  std::vector<float> aspect_ratios_vec;
  ExpandAspectRatios(aspect_ratios_, is_flip_, &aspect_ratios_vec);
  int num_priors = aspect_ratios_vec.size() * min_sizes_.size();
  if (max_sizes_.size() > 0) {
    num_priors += max_sizes_.size();
  }
  int min_num = static_cast<int>(min_sizes_.size());
  int max_num = static_cast<int>(max_sizes_.size());
  int box_num = width * height * num_priors;
  const int block_size = 256;
  const int grid_size0 = (box_num + block_size - 1) / block_size;
  const int grid_size1 = (box_num * 4 + block_size - 1) / block_size;
  float step_width, step_height;
  if (step_w_ == 0 || step_h_ == 0) {
    step_width = in_width / width;
    step_height = in_height / height;
  } else {
    step_width = step_w_;
    step_height = step_h_;
  }
  float* aspect_ratios_vec_dev;
  cudaMalloc(reinterpret_cast<void**>(&aspect_ratios_vec_dev),
             aspect_ratios_vec.size() * sizeof(float));
  cudaMemcpy(aspect_ratios_vec_dev,
             aspect_ratios_vec.data(),
             aspect_ratios_vec.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  float* min_sizes_dev;
  cudaMalloc(reinterpret_cast<void**>(&min_sizes_dev),
             min_sizes_.size() * sizeof(float));
  cudaMemcpy(min_sizes_dev,
             min_sizes_.data(),
             min_sizes_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  float* max_sizes_dev;
  cudaMalloc(reinterpret_cast<void**>(&max_sizes_dev),
             max_sizes_.size() * sizeof(float));
  cudaMemcpy(max_sizes_dev,
             max_sizes_.data(),
             max_sizes_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  float* variances_dev;
  cudaMalloc(reinterpret_cast<void**>(&variances_dev),
             variances_.size() * sizeof(float));
  cudaMemcpy(variances_dev,
             variances_.data(),
             variances_.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  float* output0 = static_cast<float*>(outputs[0]);
  prior_box_kernel<float, block_size><<<grid_size0, block_size, 0, stream>>>(
      output0,
      aspect_ratios_vec_dev,
      height,
      width,
      in_height,
      in_width,
      aspect_ratios_vec.size(),
      offset_,
      step_width,
      step_height,
      min_sizes_dev,
      max_sizes_dev,
      max_num,
      min_num,
      is_clip_,
      min_max_aspect_ratios_order_);
  float* output1 = static_cast<float*>(outputs[1]);
  SetVariance<float, block_size><<<grid_size1, block_size, 0, stream>>>(
      output1, variances_dev, static_cast<int>(variances_.size()), box_num * 4);
  cudaFree(aspect_ratios_vec_dev);
  cudaFree(min_sizes_dev);
  cudaFree(max_sizes_dev);
  cudaFree(variances_dev);
  return 0;
}

size_t PriorBoxPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(aspect_ratios_) + SerializedSize(input_dimension_) +
         SerializedSize(image_dimension_) + SerializedSize(step_w_) +
         SerializedSize(step_h_) + SerializedSize(min_sizes_) +
         SerializedSize(max_sizes_) + SerializedSize(offset_) +
         SerializedSize(is_clip_) + SerializedSize(is_flip_) +
         SerializedSize(min_max_aspect_ratios_order_) +
         SerializedSize(variances_);
}

int32_t PriorBoxPluginDynamic::getNbOutputs() const TRT_NOEXCEPT { return 2; }

nvinfer1::DataType PriorBoxPluginDynamic::getOutputDataType(
    int32_t index,
    const nvinfer1::DataType* input_types,
    int32_t nb_inputs) const TRT_NOEXCEPT {
  return input_types[0];
}

void PriorBoxPluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT {
  Serialize(&buffer, aspect_ratios_);
  Serialize(&buffer, input_dimension_);
  Serialize(&buffer, image_dimension_);
  Serialize(&buffer, step_w_);
  Serialize(&buffer, step_h_);
  Serialize(&buffer, min_sizes_);
  Serialize(&buffer, max_sizes_);
  Serialize(&buffer, offset_);
  Serialize(&buffer, is_clip_);
  Serialize(&buffer, is_flip_);
  Serialize(&buffer, min_max_aspect_ratios_order_);
  Serialize(&buffer, variances_);
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(PriorBoxPluginDynamic,
                                   PriorBoxPluginDynamicCreator,
                                   "prior_box_plugin_dynamic");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
