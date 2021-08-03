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
#include "core/hal/types.h"
#include "driver/huawei_ascend_npu/model_client.h"
#include "ge/ge_api_types.h"
#include "ge/ge_ir_build.h"
#include "graph/ge_error_codes.h"
#include "graph/graph.h"
#include "utility/logging.h"

namespace nnadapter {
namespace huawei_ascend_npu {

// Initialize/Finalize the system and allocate/free resources
void InitializeAscendDevice();
void FinalizeAscendDevice();

// Utility of the calling and error handling of Ascend ATC and ACL APIs
const std::string ACLErrorToString(int error);
const std::string ATCErrorToString(uint32_t error);
#define ACL_CALL(msg)                                                      \
  NNADAPTER_CHECK_EQ(reinterpret_cast<aclError>(msg), ACL_ERROR_NONE)      \
      << (msg) << " " << ::nnadapter::huawei_ascend_npu::ACLErrorToString( \
                             reinterpret_cast<int>(msg))
#define ATC_CALL(msg)                                                      \
  NNADAPTER_CHECK_EQ(reinterpret_cast<ge::graphStatus>(msg),               \
                     ge::GRAPH_SUCCESS)                                    \
      << (msg) << " " << ::nnadapter::huawei_ascend_npu::ATCErrorToString( \
                             reinterpret_cast<uint32_t>(msg))

// Build and load OM model to/from memory
std::shared_ptr<AclModelClient> LoadOMModelFromBuffer(
    const std::vector<uint8_t>& model_buffer, int device_id = 0);
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
  return ge::DT_UNDEFINED;
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

// Convert ACL types to GE types
ge::DataType ConvertACLDataType(aclDataType data_type);
ge::Format ConvertACLFormat(aclFormat format);
std::vector<int64_t> ConvertACLDimensions(aclmdlIODims dims);

// Convert NNAdapter types to GE types
ge::DataType ConvertPrecision(NNAdapterOperandPrecisionCode input_precision);
ge::Format ConvertDataLayout(NNAdapterOperandLayoutCode input_layout);
std::vector<int64_t> ConvertDimensions(const int32_t* input_dimensions,
                                       uint32_t input_dimensions_count);
std::vector<int64_t> ConvertDimensions(
    const std::vector<int32_t>& input_dimensions);
int32_t ConvertFuseCode(int32_t input_fuse_code);

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
