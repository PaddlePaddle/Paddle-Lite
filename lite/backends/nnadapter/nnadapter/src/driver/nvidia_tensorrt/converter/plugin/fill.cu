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

#include "driver/nvidia_tensorrt/converter/plugin/fill.h"

namespace nnadapter {
namespace nvidia_tensorrt {

FillPluginDynamic::FillPluginDynamic(float value,
                                     bool is_value_tensor,
                                     std::vector<int32_t> shape)
    : value_(value), is_value_tensor_(is_value_tensor), shape_(shape) {}

FillPluginDynamic::FillPluginDynamic(const void* serial_data,
                                     size_t serial_length) {
  Deserialize(&serial_data, &serial_length, &value_);
  Deserialize(&serial_data, &serial_length, &is_value_tensor_);
  Deserialize(&serial_data, &serial_length, &shape_);
}

nvinfer1::IPluginV2DynamicExt* FillPluginDynamic::clone() const TRT_NOEXCEPT {
  return new FillPluginDynamic(value_, is_value_tensor_, shape_);
}

nvinfer1::DimsExprs FillPluginDynamic::getOutputDimensions(
    int32_t output_index,
    const nvinfer1::DimsExprs* inputs,
    int32_t nb_inputs,
    nvinfer1::IExprBuilder& expr_builder) TRT_NOEXCEPT {
  NNADAPTER_CHECK_EQ(output_index, 0);
  NNADAPTER_CHECK(inputs);
  NNADAPTER_CHECK_GE(nb_inputs, 1);
  nvinfer1::DimsExprs outdims;
  outdims.nbDims = shape_.size();
  for (int i = 0; i < shape_.size(); i++) {
    outdims.d[i] = expr_builder.constant(shape_[i]);
  }
  return outdims;
}

template <typename T, unsigned TPB>
__global__ void fill_kernel_value(int n, T* output, T value) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < n) {
    output[idx] = value;
  }
}

template <typename T, unsigned TPB>
__global__ void fill_kernel_value_tensor(int n, T* output, const T* value) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < n) {
    output[idx] = *value;
  }
}

int32_t FillPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto output_dims = output_desc[0].dims;
  int num = 1;
  for (int i = 0; i < output_dims.nbDims; i++) {
    num *= output_dims.d[i];
  }
  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;

  float* output = static_cast<float*>(outputs[0]);
  if (is_value_tensor_)
    fill_kernel_value_tensor<float,
                             block_size><<<grid_size, block_size, 0, stream>>>(
        num, output, (static_cast<const float*>(inputs[0])));
  else
    fill_kernel_value<float, block_size><<<grid_size, block_size, 0, stream>>>(
        num, output, value_);

  return 0;
}

size_t FillPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(value_) + SerializedSize(is_value_tensor_) +
         SerializedSize(shape_);
}

void FillPluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT {
  Serialize(&buffer, value_);
  Serialize(&buffer, is_value_tensor_);
  Serialize(&buffer, shape_);
}

bool FillPluginDynamic::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) TRT_NOEXCEPT {
  NNADAPTER_CHECK_LT(pos, nb_inputs + nb_outputs);
  NNADAPTER_CHECK(in_out);
  return true;
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(FillPluginDynamic,
                                   FillPluginDynamicCreator,
                                   "fill_plugin_dynamic");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
