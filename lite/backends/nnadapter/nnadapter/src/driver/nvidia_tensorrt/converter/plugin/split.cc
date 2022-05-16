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

#include "driver/nvidia_tensorrt/converter/plugin/split.h"

namespace nnadapter {
namespace nvidia_tensorrt {

int SplitPlugin::initialize() TRT_NOEXCEPT {
  // notice input dims is [C, H, W]
  nvinfer1::Dims dims = input_dims_[0];
  outer_rows_ = 1;
  inner_cols_ = 1;
  for (int i = 0; i < axis_; ++i) {
    outer_rows_ *= dims.d[i];
  }
  for (int i = axis_ + 1; i < dims.nbDims; ++i) {
    inner_cols_ *= dims.d[i];
  }
  std::vector<int> segment_offsets(1, 0);
  for (int i = 0; i < this->getNbOutputs(); ++i) {
    segment_offsets.push_back(segment_offsets.back() + size_splits_[i]);
  }
  axis_shape_ = dims.d[axis_];
  cudaMalloc(reinterpret_cast<void**>(&dev_segment_offsets_),
             segment_offsets.size() * sizeof(int));
  cudaMemcpy(dev_segment_offsets_,
             segment_offsets.data(),
             segment_offsets.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  return 0;
}

void SplitPlugin::terminate() TRT_NOEXCEPT {
  cudaFree(dev_segment_offsets_);
  cudaFree(dev_output_ptrs_);
}

int SplitPlugin::enqueue(int batch_size,
#if TENSORRT_VERSION_GE(8, 0, 0, 0)
                         void const* const* inputs,
                         void* const* outputs,
#else
                         const void* const* inputs,
                         void** outputs,
#endif
                         void* workspace,
                         cudaStream_t stream) TRT_NOEXCEPT {
  float const* input_ptr = reinterpret_cast<float const*>(inputs[0]);
  float* const* outputs_ptr = reinterpret_cast<float* const*>(outputs);
  cudaMalloc(reinterpret_cast<void**>(&dev_output_ptrs_),
             size_splits_.size() * sizeof(float*));
  cudaMemcpyAsync(dev_output_ptrs_,
                  outputs_ptr,
                  size_splits_.size() * sizeof(float*),
                  cudaMemcpyHostToDevice,
                  stream);
  Split(input_ptr,
        dev_output_ptrs_,
        dev_segment_offsets_,
        size_splits_.size(),
        inner_cols_,
        axis_shape_,
        outer_rows_,
        batch_size,
        stream);
  return 0;
}

nvinfer1::Dims SplitPlugin::getOutputDimensions(
    int index, const nvinfer1::Dims* inputs, int nb_input_dims) TRT_NOEXCEPT {
  nvinfer1::Dims output_dims = inputs[0];
  output_dims.d[axis_] = size_splits_.at(index);
  return output_dims;
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(SplitPlugin,
                                   SplitPluginCreator,
                                   "split_plugin");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
