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
#include "core/types.h"
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

// Specify the file path to dump the model
#define HUAWEI_ASCEND_NPU_DUMP_MODEL_FILE_PATH \
  "HUAWEI_ASCEND_NPU_DUMP_MODEL_FILE_PATH"

// Select operator precision mode
#define HUAWEI_ASCEND_NPU_PRECISION_MODE "HUAWEI_ASCEND_NPU_PRECISION_MODE"

// Specify the file path of the modify mixlist if precision mode is
// allow_mix_precision
#define HUAWEI_ASCEND_NPU_MODIFY_MIXLIST_FILE_PATH \
  "HUAWEI_ASCEND_NPU_MODIFY_MIXLIST_FILE_PATH"

// Specify OP_SELECT_IMPL_MODE
#define HUAWEI_ASCEND_NPU_OP_SELECT_IMPL_MODE \
  "HUAWEI_ASCEND_NPU_OP_SELECT_IMPL_MODE"

// Specify OPTYPELIST_FOR_IMPLMODE
#define HUAWEI_ASCEND_NPU_OPTYPELIST_FOR_IMPLMODE \
  "HUAWEI_ASCEND_NPU_OPTYPELIST_FOR_IMPLMODE"

// Specify ENABLE_COMPRESS_WEIGHT
#define HUAWEI_ASCEND_NPU_ENABLE_COMPRESS_WEIGHT \
  "HUAWEI_ASCEND_NPU_ENABLE_COMPRESS_WEIGHT"

// Specify AUTO_TUNE_MODE
#define HUAWEI_ASCEND_NPU_AUTO_TUNE_MODE "HUAWEI_ASCEND_NPU_AUTO_TUNE_MODE"

// Specify ENABLE_DYNAMIC_SHAPE_RANGE
#define HUAWEI_ASCEND_NPU_ENABLE_DYNAMIC_SHAPE_RANGE \
  "HUAWEI_ASCEND_NPU_ENABLE_DYNAMIC_SHAPE_RANGE"

// Specify the buffer length initialized of dynamic_shape_range
#define HUAWEI_ASCEND_NPU_INITIAL_BUFFER_LENGTH_OF_DYNAMIC_SHAPE_RANGE \
  "HUAWEI_ASCEND_NPU_INITIAL_BUFFER_LENGTH_OF_DYNAMIC_SHAPE_RANGE"

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
void InitializeGraphBuilder(AscendConfigParams* config_params);

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
    AscendConfigParams* config_params);
bool BuildOMModelToBuffer(
    std::vector<ge::Operator>& input_operators,   // NOLINT
    std::vector<ge::Operator>& output_operators,  // NOLINT
    std::vector<uint8_t>* model_buffer,
    const std::vector<std::string>& dynamic_shape_info,
    const std::string& optional_shape_str,
    const DynamicShapeMode dynamic_shape_mode,
    AscendConfigParams* config_params);

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

// Generate shape strings for CANN
std::string ShapeToString(const std::vector<int32_t>& shape);
std::string MergeOptionalShapInfo(
    const std::vector<std::string>& optional_shape_info,
    const DynamicShapeMode dynamic_shape_mode);
void GetDynamicShapeInfo(const std::vector<NNAdapterOperandType>& input_types,
                         std::vector<std::string>* dynamic_shape_info,
                         std::string* optional_shape_str,
                         DynamicShapeMode* dynamic_shape_mode);

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
