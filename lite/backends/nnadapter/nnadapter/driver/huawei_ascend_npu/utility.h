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
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {
namespace huawei_ascend_npu {

// The following environment variables can be used at runtime:
// Specify the list of device IDs, such as
// HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0,1,2,3 or
// HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS=0
#define HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS \
  "HUAWEI_ASCEND_NPU_SELECTED_DEVICE_IDS"

// Specify the file path of the profiling results
#define HUAWEI_ASCEND_NPU_PROFILING_FILE_PATH \
  "HUAWEI_ASCEND_NPU_PROFILING_FILE_PATH"

#define NNADAPTER_HUAWEI_ASCEND_NPU_CANN_VERSION_GREATER_THAN(   \
    major, minor, patch)                                         \
  NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MAJOR_VERSION * 1000 +        \
          NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MINOR_VERSION * 100 + \
          NNADAPTER_HUAWEI_ASCEND_NPU_CANN_PATCH_VERSION >=      \
      major * 1000 + minor * 100 + patch

#define NNADAPTER_HUAWEI_ASCEND_NPU_CANN_VERSION_LESS_THAN(      \
    major, minor, patch)                                         \
  NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MAJOR_VERSION * 1000 +        \
          NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MINOR_VERSION * 100 + \
          NNADAPTER_HUAWEI_ASCEND_NPU_CANN_PATCH_VERSION <       \
      major * 1000 + minor * 100 + patch

// Prepare AscendCL environment and register the finalizer to be called at
// normal process termination
void InitializeAscendCL();
// Initialize the resources of the model builder and register the finalizer to
// be called at normal process termination
void InitializeGraphBuilder();

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
    const std::vector<uint8_t>& model_buffer,
    int device_id,
    const std::string& profiling_file_path);
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
ge::DataType ConvertACLDataTypeToGEDataType(aclDataType input_data_type);
ge::Format ConvertACLFormatToGEFormat(aclFormat input_format);
std::vector<int64_t> ConvertACLDimsToGEDims(
    const aclmdlIODims& input_dimensions);
void ConvertACLDimsToGEDims(const aclmdlIODims& input_dimensions,
                            int32_t* output_dimensions,
                            uint32_t* output_dimensions_count);

// Convert NNAdapter types to GE types
ge::DataType ConvertToGEPrecision(NNAdapterOperandPrecisionCode precision_code);
ge::Format ConvertToGEDataLayout(NNAdapterOperandLayoutCode layout_code);
std::vector<int64_t> ConvertToGEDimensions(const int32_t* input_dimensions,
                                           uint32_t input_dimensions_count);
std::vector<int64_t> ConvertToGEDimensions(
    const std::vector<int32_t>& input_dimensions);
std::string ConvertPadModeCodeToGEPadMode(int pad_mode_code);
std::string ConvertInterpolateModeCodeToGEInterpolateMode(
    int interpolate_mode_code);

// Get Ascend CANN version
bool GetAscendCANNVersion(int* major, int* minor, int* patch);

// Get Ascend soc name
ge::AscendString GetAscendSocName();

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
