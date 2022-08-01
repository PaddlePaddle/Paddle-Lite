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

#include <memory>
#include <string>
#include <vector>
#include "HiAiModelManagerService.h"  // NOLINT
#include "core/types.h"
#include "hiai_ir_build.h"  // NOLINT
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_kirin_npu {

// Build and load OM model to/from memory
std::shared_ptr<hiai::AiModelMngerClient> LoadOMModelFromBuffer(
    const std::string& model_name,
    std::vector<uint8_t>* model_buffer,
    bool* model_comp,
    int freq_level,
    int framework_type,
    int model_type,
    int device_type);
bool BuildOMModelToBuffer(
    std::vector<ge::Operator>& input_operators,   // NOLINT
    std::vector<ge::Operator>& output_operators,  // NOLINT
    std::vector<uint8_t>* model_buffer);

// Convert GE types to strings
const std::string GEDataTypeToString(ge::DataType data_type);
const std::string GEFormatToString(ge::Format format);
const std::string GEShapeToString(ge::Shape shape);
int64_t ProductionOfGEShape(ge::Shape shape);

// Convert C/C++ POD types to GE data type
template <typename T>
ge::DataType GetGEDataType() {
  NNADAPTER_LOG(FATAL) << "Unable to convert " << typeid(T).name()
                       << " to ge::DataType";
}
template <>
ge::DataType GetGEDataType<float>();
template <>
ge::DataType GetGEDataType<int8_t>();
template <>
ge::DataType GetGEDataType<int16_t>();
template <>
ge::DataType GetGEDataType<int32_t>();
template <>
ge::DataType GetGEDataType<int64_t>();
template <>
ge::DataType GetGEDataType<bool>();

// Convert NNAdapter types to GE types
ge::DataType ConvertToGEPrecision(NNAdapterOperandPrecisionCode precision_code);
ge::Format ConvertToGEDataLayout(NNAdapterOperandLayoutCode layout_code);
std::vector<int64_t> ConvertToGEDimensions(const int32_t* input_dimensions,
                                           uint32_t input_dimensions_count);
std::vector<int64_t> ConvertToGEDimensions(
    const std::vector<int32_t>& input_dimensions);
int32_t ConvertFuseCodeToGEActMode(int32_t fuse_code);
std::string ConvertAutoPadCodeToGEPadMode(NNAdapterAutoPadCode auto_pad_code);
std::string ConvertPadModeCodeToGEPadMode(int pad_mode_code);

}  // namespace huawei_kirin_npu
}  // namespace nnadapter
