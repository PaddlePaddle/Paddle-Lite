// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <xtcl/xtcl.h>
#include <memory>
#include <string>
#include <vector>
#include "core/types.h"
#include "utility/logging.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

// The following environment variables can be used at runtime:
// Specify the list of device IDs, such as
// KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS=0,1,2,3 or
// KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS=0
#define KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS "KUNLUNXIN_XTCL_SELECTED_DEVICE_IDS"
// Specify the target type when the model is compiled, which supports 'llvm' and
// 'xpu-libs=xdnn', the default value is empty.
#define KUNLUNXIN_XTCL_DEVICE_TARGET "KUNLUNXIN_XTCL_DEVICE_TARGET"

// Build and load XTCL runtime instance(compiled XTCL model) to/from memory
std::shared_ptr<xtcl::network::xRuntimeInstance> LoadInstanceRuntimeFromBuffer(
    int device_id, std::vector<uint8_t>* model_buffer);
std::shared_ptr<xtcl::network::xRuntimeInstance> BuildInstanceRuntimeToBuffer(
    int device_id,
    std::string device_target,
    xtcl::network::xNetworkBuilder* builder,
    xtcl::network::xTensorCompiler::ParamNDArrayMap* params,
    xtcl::Array<xtcl::xExpr>* outputs,
    std::vector<uint8_t>* model_buffer);

xtcl::DataType ConvertToXTCLDataType(
    NNAdapterOperandPrecisionCode input_precision);
DLDataType ConvertToDLDataType(NNAdapterOperandPrecisionCode input_precision);

template <typename T>
xtcl::Array<T> ConvertToXTCLArray(const int32_t* input_dimensions,
                                  uint32_t input_dimensions_count) {
  xtcl::Array<T> output_dimensions;
  for (uint32_t i = 0; i < input_dimensions_count; i++) {
    output_dimensions.push_back(input_dimensions[i]);
  }
  return output_dimensions;
}

template <typename T>
xtcl::Array<T> ConvertToXTCLArray(
    const std::vector<int32_t>& input_dimensions) {
  return ConvertToXTCLArray<T>(input_dimensions.data(),
                               input_dimensions.size());
}

xtcl::xNDArray CreateXTCLNDArray(std::vector<int64_t> shape,
                                 DLDataType dtype,
                                 const void* buffer);

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
