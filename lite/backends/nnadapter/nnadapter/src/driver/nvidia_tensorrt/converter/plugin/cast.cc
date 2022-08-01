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

CastPlugin::CastPlugin() {}

CastPlugin::CastPlugin(nvinfer1::DataType intype, nvinfer1::DataType outtype)
    : intype_(intype), outtype_(outtype) {}

CastPlugin::CastPlugin(const void* serial_data, size_t serial_length) {
  int intype, outtype;
  Deserialize(&serial_data, &serial_length, &intype);
  Deserialize(&serial_data, &serial_length, &outtype);
  intype_ = (nvinfer1::DataType)(intype);
  outtype_ = (nvinfer1::DataType)(outtype);
}

CastPlugin::~CastPlugin() TRT_NOEXCEPT {}

bool CastPlugin::supportsFormat(
    nvinfer1::DataType type, nvinfer1::PluginFormat format) const TRT_NOEXCEPT {
  return type == nvinfer1::DataType::kFLOAT ||
         type == nvinfer1::DataType::kINT32;
}

int CastPlugin::enqueue(int batch_size,
#if TENSORRT_VERSION_GE(8, 0, 0, 0)
                        void const* const* inputs,
                        void* const* outputs,
#else
                        const void* const* inputs,
                        void** outputs,
#endif
                        void* workspace,
                        cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_dims_[0];
  int num = batch_size;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  if (intype_ == nvinfer1::DataType::kINT32 &&
      outtype_ == nvinfer1::DataType::kFLOAT) {  // int32->float32
    const int32_t* input = static_cast<const int32_t*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    NNADAPTER_CHECK_EQ((Cast<int32_t, float>(input, output, num, stream)),
                       cudaSuccess);
  } else if (intype_ == nvinfer1::DataType::kFLOAT &&
             outtype_ == nvinfer1::DataType::kINT32) {  // float32->int32
    const float* input = static_cast<const float*>(inputs[0]);
    int32_t* output = static_cast<int32_t*>(outputs[0]);
    NNADAPTER_CHECK_EQ((Cast<float, int32_t>(input, output, num, stream)),
                       cudaSuccess);
  } else {
    NNADAPTER_LOG(FATAL) << "cast nvidia_tensorrt doesn't support this cast";
  }
  return 0;
}

size_t CastPlugin::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(static_cast<int>(outtype_)) +
         SerializedSize(static_cast<int>(intype_));
}

void CastPlugin::serialize(void* buffer) const TRT_NOEXCEPT {
  Serialize(&buffer, static_cast<int>(outtype_));
  Serialize(&buffer, static_cast<int>(intype_));
}

nvinfer1::IPluginV2* CastPlugin::clone() const TRT_NOEXCEPT {
  return new CastPlugin(intype_, outtype_);
}

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

nvinfer1::IPluginV2DynamicExt* CastPluginDynamic::clone() const TRT_NOEXCEPT {
  return new CastPluginDynamic(intype_, outtype_);
}

int32_t CastPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    void const* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) TRT_NOEXCEPT {
  auto input_dims = input_desc[0].dims;
  int num = 1;
  for (int i = 0; i < input_dims.nbDims; i++) {
    num *= input_dims.d[i];
  }
  if (intype_ == nvinfer1::DataType::kINT32 &&
      outtype_ == nvinfer1::DataType::kFLOAT) {  // int32->float32
    const int32_t* input = static_cast<const int32_t*>(inputs[0]);
    float* output = static_cast<float*>(outputs[0]);
    NNADAPTER_CHECK_EQ((Cast<int32_t, float>(input, output, num, stream)),
                       cudaSuccess);
  } else if (intype_ == nvinfer1::DataType::kFLOAT &&
             outtype_ == nvinfer1::DataType::kINT32) {  // float32->int32
    const float* input = static_cast<const float*>(inputs[0]);
    int32_t* output = static_cast<int32_t*>(outputs[0]);
    NNADAPTER_CHECK_EQ((Cast<float, int32_t>(input, output, num, stream)),
                       cudaSuccess);
  } else {
    NNADAPTER_LOG(FATAL) << "cast nvidia_tensorrt doesn't support this cast";
  }
  return 0;
}

size_t CastPluginDynamic::getSerializationSize() const TRT_NOEXCEPT {
  return SerializedSize(static_cast<int>(outtype_)) +
         SerializedSize(static_cast<int>(intype_));
}

void CastPluginDynamic::serialize(void* buffer) const TRT_NOEXCEPT {
  Serialize(&buffer, static_cast<int>(outtype_));
  Serialize(&buffer, static_cast<int>(intype_));
}

bool CastPluginDynamic::supportsFormatCombination(
    int32_t pos,
    const nvinfer1::PluginTensorDesc* in_out,
    int32_t nb_inputs,
    int32_t nb_outputs) TRT_NOEXCEPT {
  NNADAPTER_CHECK_LT(pos, nb_inputs + nb_outputs);
  NNADAPTER_CHECK(in_out);
  return true;
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(CastPlugin,
                                   CastPluginCreator,
                                   "cast_plugin");

REGISTER_NNADAPTER_TENSORRT_PLUGIN(CastPluginDynamic,
                                   CastPluginDynamicCreator,
                                   "cast_plugin_dynamic");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
