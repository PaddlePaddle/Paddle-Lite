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

#include "driver/nvidia_tensorrt/converter/plugin/cast.h"

namespace nnadapter {
namespace nvidia_tensorrt {

CastPluginDynamic::CastPluginDynamic(nvinfer1::DataType intype,
                                     nvinfer1::DataType outtype)
    : intype_(intype), outtype_(outtype) {}

CastPluginDynamic::CastPluginDynamic(const void* serial_data,
                                     size_t serial_length) {
  int intype, outtype;
  Deserialize(&serial_data, &serial_length, &intype);
  Deserialize(&serial_data, &serial_length, &outtype);
  intype_ = (nvinfer1::DataType)(intype);
  outtype_ = (nvinfer1::DataType)(outtype);
}

nvinfer1::IPluginV2DynamicExt* CastPluginDynamic::clone() const noexcept {
  return new CastPluginDynamic(intype_, outtype_);
}

template <typename Tin, typename Tout, unsigned TPB>
__global__ void cast_kernel(int n, const Tin* input, Tout* output) {
  const int idx = blockIdx.x * TPB + threadIdx.x;
  if (idx < n) {
    output[idx] = static_cast<Tout>(input[idx]);
  }
}

int32_t CastPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  auto input_dims = input_desc[0].dims;
  int num = 1;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  const int block_size = 256;
  const int grid_size = (num + block_size - 1) / block_size;

  if (intype_ == nvinfer1::DataType::kINT32 &&
      outtype_ == nvinfer1::DataType::kFLOAT) {  // int32->float32
    const int32_t* input = static_cast<const int32_t*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    cast_kernel<int32_t,
                float,
                block_size><<<grid_size, block_size, 0, stream>>>(
        num, input, output);
  } else if (intype_ == nvinfer1::DataType::kFLOAT &&
             outtype_ == nvinfer1::DataType::kINT32) {  // float32->int32
    const float* input = static_cast<const float*>(inputs[0]);
    int32_t* output = static_cast<int32_t*>(outputs[0]);
    cast_kernel<float,
                int32_t,
                block_size><<<grid_size, block_size, 0, stream>>>(
        num, input, output);
  } else {
    NNADAPTER_LOG(FATAL) << "cast nvidia_tensorrt doesn't support this cast";
  }

  return 0;
}

size_t CastPluginDynamic::getSerializationSize() const noexcept {
  return SerializedSize((int)(outtype_)) + SerializedSize((int)(intype_));
}

void CastPluginDynamic::serialize(void* buffer) const noexcept {
  Serialize(&buffer, (int)(outtype_));
  Serialize(&buffer, (int)(intype_));
}

bool CastPluginDynamic::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) noexcept {
  NNADAPTER_CHECK_LT(pos, nb_inputs + nb_outputs);
  NNADAPTER_CHECK(in_out);
  return true;
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(CastPluginDynamic,
                                   CastPluginDynamicCreator,
                                   "cast_plugin_dynamic");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
