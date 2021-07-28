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

#include "driver/huawei_ascend_npu/utility.h"
#include <map>
#include <utility>
#include "utility/debug.h"
#include "utility/string.h"

namespace nnadapter {
namespace huawei_ascend_npu {

void InitializeAscendDevice() {
  NNADAPTER_VLOG(5) << "Intialize the system and allocate resources.";
  // The following APIs can only be called once in one process
  aclInit(NULL);
  std::map<ge::AscendString, ge::AscendString> global_options;
  global_options.insert(
      std::make_pair(ge::ir_option::SOC_VERSION, "Ascend310"));
  ge::aclgrphBuildInitialize(global_options);
}

void FinalizeAscendDevice() {
  NNADAPTER_VLOG(5) << "Release the resources.";
  // The following APIs can only be called once in one process
  ge::aclgrphBuildFinalize();
  aclFinalize();
}

const std::string ACLErrorToString(int error) {
  switch (error) {
#define ACL_ERROR_TO_STRING(error) \
  case error:                      \
    return std::string(#error);
    ACL_ERROR_TO_STRING(ACL_ERROR_INVALID_PARAM);                    // 100000
    ACL_ERROR_TO_STRING(ACL_ERROR_UNINITIALIZE);                     // 100001
    ACL_ERROR_TO_STRING(ACL_ERROR_REPEAT_INITIALIZE);                // 100002
    ACL_ERROR_TO_STRING(ACL_ERROR_INVALID_FILE);                     // 100003
    ACL_ERROR_TO_STRING(ACL_ERROR_WRITE_FILE);                       // 100004
    ACL_ERROR_TO_STRING(ACL_ERROR_INVALID_FILE_SIZE);                // 100005
    ACL_ERROR_TO_STRING(ACL_ERROR_PARSE_FILE);                       // 100006
    ACL_ERROR_TO_STRING(ACL_ERROR_FILE_MISSING_ATTR);                // 100007
    ACL_ERROR_TO_STRING(ACL_ERROR_FILE_ATTR_INVALID);                // 100008
    ACL_ERROR_TO_STRING(ACL_ERROR_INVALID_DUMP_CONFIG);              // 100009
    ACL_ERROR_TO_STRING(ACL_ERROR_INVALID_PROFILING_CONFIG);         // 100010
    ACL_ERROR_TO_STRING(ACL_ERROR_INVALID_MODEL_ID);                 // 100011
    ACL_ERROR_TO_STRING(ACL_ERROR_DESERIALIZE_MODEL);                // 100012
    ACL_ERROR_TO_STRING(ACL_ERROR_PARSE_MODEL);                      // 100013
    ACL_ERROR_TO_STRING(ACL_ERROR_READ_MODEL_FAILURE);               // 100014
    ACL_ERROR_TO_STRING(ACL_ERROR_MODEL_SIZE_INVALID);               // 100015
    ACL_ERROR_TO_STRING(ACL_ERROR_MODEL_MISSING_ATTR);               // 100016
    ACL_ERROR_TO_STRING(ACL_ERROR_MODEL_INPUT_NOT_MATCH);            // 100017
    ACL_ERROR_TO_STRING(ACL_ERROR_MODEL_OUTPUT_NOT_MATCH);           // 100018
    ACL_ERROR_TO_STRING(ACL_ERROR_MODEL_NOT_DYNAMIC);                // 100019
    ACL_ERROR_TO_STRING(ACL_ERROR_OP_TYPE_NOT_MATCH);                // 100020
    ACL_ERROR_TO_STRING(ACL_ERROR_OP_INPUT_NOT_MATCH);               // 100021
    ACL_ERROR_TO_STRING(ACL_ERROR_OP_OUTPUT_NOT_MATCH);              // 100022
    ACL_ERROR_TO_STRING(ACL_ERROR_OP_ATTR_NOT_MATCH);                // 100023
    ACL_ERROR_TO_STRING(ACL_ERROR_OP_NOT_FOUND);                     // 100024
    ACL_ERROR_TO_STRING(ACL_ERROR_OP_LOAD_FAILED);                   // 100025
    ACL_ERROR_TO_STRING(ACL_ERROR_UNSUPPORTED_DATA_TYPE);            // 100026
    ACL_ERROR_TO_STRING(ACL_ERROR_FORMAT_NOT_MATCH);                 // 100027
    ACL_ERROR_TO_STRING(ACL_ERROR_BIN_SELECTOR_NOT_REGISTERED);      // 100028
    ACL_ERROR_TO_STRING(ACL_ERROR_KERNEL_NOT_FOUND);                 // 100029
    ACL_ERROR_TO_STRING(ACL_ERROR_BIN_SELECTOR_ALREADY_REGISTERED);  // 100030
    ACL_ERROR_TO_STRING(ACL_ERROR_KERNEL_ALREADY_REGISTERED);        // 100031
    ACL_ERROR_TO_STRING(ACL_ERROR_INVALID_QUEUE_ID);                 // 100032
    ACL_ERROR_TO_STRING(ACL_ERROR_REPEAT_SUBSCRIBE);                 // 100033
    ACL_ERROR_TO_STRING(ACL_ERROR_STREAM_NOT_SUBSCRIBE);             // 100034
    ACL_ERROR_TO_STRING(ACL_ERROR_THREAD_NOT_SUBSCRIBE);             // 100035
    ACL_ERROR_TO_STRING(ACL_ERROR_WAIT_CALLBACK_TIMEOUT);            // 100036
    ACL_ERROR_TO_STRING(ACL_ERROR_REPEAT_FINALIZE);                  // 100037
    ACL_ERROR_TO_STRING(ACL_ERROR_NOT_STATIC_AIPP);                  // 100038
    ACL_ERROR_TO_STRING(ACL_ERROR_BAD_ALLOC);                        // 200000
    ACL_ERROR_TO_STRING(ACL_ERROR_API_NOT_SUPPORT);                  // 200001
    ACL_ERROR_TO_STRING(ACL_ERROR_INVALID_DEVICE);                   // 200002
    ACL_ERROR_TO_STRING(ACL_ERROR_MEMORY_ADDRESS_UNALIGNED);         // 200003
    ACL_ERROR_TO_STRING(ACL_ERROR_RESOURCE_NOT_MATCH);               // 200004
    ACL_ERROR_TO_STRING(ACL_ERROR_INVALID_RESOURCE_HANDLE);          // 200005
    ACL_ERROR_TO_STRING(ACL_ERROR_FEATURE_UNSUPPORTED);              // 200006
    ACL_ERROR_TO_STRING(ACL_ERROR_STORAGE_OVER_LIMIT);               // 300000
    ACL_ERROR_TO_STRING(ACL_ERROR_INTERNAL_ERROR);                   // 500000
    ACL_ERROR_TO_STRING(ACL_ERROR_FAILURE);                          // 500001
    ACL_ERROR_TO_STRING(ACL_ERROR_GE_FAILURE);                       // 500002
    ACL_ERROR_TO_STRING(ACL_ERROR_RT_FAILURE);                       // 500003
    ACL_ERROR_TO_STRING(ACL_ERROR_DRV_FAILURE);                      // 500004
    ACL_ERROR_TO_STRING(ACL_ERROR_PROFILING_FAILURE);                // 500005
#undef ACL_ERROR_TO_STRING
    default:
      return string_format("Unknown ACL error code(%d)", error).c_str();
  }
}

const std::string ATCErrorToString(uint32_t error) {
  switch (error) {
#define ATC_ERROR_TO_STRING(error) \
  case error:                      \
    return std::string(#error);
    ATC_ERROR_TO_STRING(ge::GRAPH_FAILED);                    // 0xFFFFFFFF
    ATC_ERROR_TO_STRING(ge::GRAPH_NOT_CHANGED);               // 1343242304
    ATC_ERROR_TO_STRING(ge::GRAPH_PARAM_INVALID);             // 50331649
    ATC_ERROR_TO_STRING(ge::GRAPH_NODE_WITHOUT_CONST_INPUT);  // 50331648
#undef ATC_ERROR_TO_STRING
    default:
      return string_format("Unknown ATC error code(%d)", error).c_str();
  }
}

std::shared_ptr<AclModelClient> LoadOMModelFromBuffer(
    const std::vector<uint8_t>& model_buffer, int device_id) {
  if (model_buffer.size() == 0) {
    NNADAPTER_LOG(ERROR) << "model_buffer size should not be 0!";
    return nullptr;
  }
  // Create a ACL model client to load the om model
  auto model_client = std::make_shared<AclModelClient>(device_id);
  // Load model from memory
  if (model_client->LoadModel(
          reinterpret_cast<const void*>(model_buffer.data()),
          model_buffer.size())) {
    return model_client;
  }
  return nullptr;
}

bool BuildOMModelToBuffer(
    std::vector<ge::Operator>& input_operators,   // NOLINT
    std::vector<ge::Operator>& output_operators,  // NOLINT
    std::vector<uint8_t>* model_buffer) {
  // Convert the CANN IR graph to the CANN om model
  ge::Graph ir_graph("graph");
  // Set input operator attr index if node size > 1
  auto input_count = input_operators.size();
  auto output_count = output_operators.size();
  NNADAPTER_VLOG(3) << "input_count: " << input_count;
  NNADAPTER_VLOG(3) << "output_count: " << output_count;
  NNADAPTER_CHECK_GE(output_count, 1);
  if (input_count > 1) {
    for (size_t i = 0; i < input_count; i++) {
      input_operators[i].SetAttr("index", static_cast<int>(i));
    }
  }
  ir_graph.SetInputs(input_operators).SetOutputs(output_operators);
  // Build IR model
  ge::ModelBufferData om_buffer;
  std::map<ge::AscendString, ge::AscendString> options;
  options.insert(std::make_pair(ge::ir_option::LOG_LEVEL, "error"));
  ATC_CALL(aclgrphBuildModel(ir_graph, options, om_buffer));
  // Copy from om model buffer
  model_buffer->resize(om_buffer.length);
  memcpy(reinterpret_cast<void*>(model_buffer->data()),
         reinterpret_cast<void*>(om_buffer.data.get()),
         om_buffer.length);
  return true;
}

const std::string GEDataTypeToString(ge::DataType data_type) {
  static const std::vector<std::string> datatype2strings{"DT_FLOAT=0",
                                                         "DT_FLOAT16=1",
                                                         "DT_INT8=2",
                                                         "DT_INT32=3",
                                                         "DT_UINT8=4",
                                                         "Unknown=5",
                                                         "DT_INT16=6",
                                                         "DT_UINT16=7",
                                                         "DT_UINT32=8",
                                                         "DT_INT64=9",
                                                         "DT_UINT64=10",
                                                         "DT_DOUBLE=11",
                                                         "DT_BOOL=12",
                                                         "DT_STRING=13"};
  auto index = static_cast<int>(data_type);
  NNADAPTER_CHECK_LT(index, datatype2strings.size());
  return datatype2strings[index];
}

const std::string GEFormatToString(ge::Format format) {
  static const std::vector<std::string> format2strings = {
      "FORMAT_NCHW = 0",
      "FORMAT_NHWC = 1",
      "FORMAT_ND = 2",
      "FORMAT_NC1HWC0 = 3",
      "FORMAT_FRACTAL_Z = 4",
      "FORMAT_NC1C0HWPAD = 5",
      "FORMAT_NHWC1C0 = 6",
      "FORMAT_FSR_NCHW = 7",
      "FORMAT_FRACTAL_DECONV = 8",
      "FORMAT_C1HWNC0 = 9",
      "FORMAT_FRACTAL_DECONV_TRANSPOSE = 10",
      "FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS = 11",
      "FORMAT_NC1HWC0_C04 = 12",
      "FORMAT_FRACTAL_Z_C04 = 13",
      "FORMAT_CHWN = 14",
      "FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS = 15",
      "FORMAT_HWCN = 16",
      "FORMAT_NC1KHKWHWC0 = 17",
      "FORMAT_BN_WEIGHT = 18",
      "FORMAT_FILTER_HWCK = 19",
      "FORMAT_HASHTABLE_LOOKUP_LOOKUPS = 20",
      "FORMAT_HASHTABLE_LOOKUP_KEYS = 21",
      "FORMAT_HASHTABLE_LOOKUP_VALUE = 22",
      "FORMAT_HASHTABLE_LOOKUP_OUTPUT = 23",
      "FORMAT_HASHTABLE_LOOKUP_HITS = 24"};
  auto index = static_cast<int>(format);
  NNADAPTER_CHECK_LT(index, format2strings.size());
  return format2strings[index];
}

const std::string GEShapeToString(ge::Shape shape) {
  std::stringstream ss;
  size_t dim_count = shape.GetDimNum();
  if (dim_count == 0) {
    ss << "{}";
    return ss.str();
  }
  ss << "{";
  for (size_t i = 0; i < dim_count - 1; i++) {
    ss << shape.GetDim(i) << ",";
  }
  ss << shape.GetDim(dim_count - 1);
  ss << "}";
  return ss.str();
}

int64_t ProductionOfGEShape(ge::Shape shape) {
  int64_t production = 1;
  size_t dim_count = shape.GetDimNum();
  for (size_t i = 0; i < dim_count; i++) {
    auto dimension = shape.GetDim(i);
    NNADAPTER_CHECK_GT(dimension, 0);
    production *= dimension;
  }
  return production;
}

template <>
ge::DataType GetGEDataType<float>() {
  return ge::DT_FLOAT;
}

template <>
ge::DataType GetGEDataType<int8_t>() {
  return ge::DT_INT8;
}

template <>
ge::DataType GetGEDataType<int16_t>() {
  return ge::DT_INT16;
}

template <>
ge::DataType GetGEDataType<int32_t>() {
  return ge::DT_INT32;
}

template <>
ge::DataType GetGEDataType<int64_t>() {
  return ge::DT_INT64;
}

template <>
ge::DataType GetGEDataType<bool>() {
  return ge::DT_BOOL;
}

ge::DataType ConvertACLDataType(aclDataType input_data_type) {
  ge::DataType output_data_type = ge::DT_FLOAT;
  switch (input_data_type) {
    case ACL_FLOAT:
      output_data_type = ge::DT_FLOAT;
      break;
    case ACL_FLOAT16:
      output_data_type = ge::DT_FLOAT16;
      break;
    case ACL_INT8:
      output_data_type = ge::DT_INT8;
      break;
    case ACL_INT16:
      output_data_type = ge::DT_INT16;
      break;
    case ACL_INT32:
      output_data_type = ge::DT_INT32;
      break;
    case ACL_INT64:
      output_data_type = ge::DT_INT64;
      break;
    case ACL_BOOL:
      output_data_type = ge::DT_BOOL;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to convert aclDataType("
                           << input_data_type << ") to ge::DataType.";
      break;
  }
  NNADAPTER_VLOG(5) << "geDataType: " << GEDataTypeToString(output_data_type);
  return output_data_type;
}

ge::Format ConvertACLFormat(aclFormat input_format) {
  ge::Format output_format = ge::FORMAT_NCHW;
  switch (input_format) {
    case ACL_FORMAT_NCHW:
      output_format = ge::FORMAT_NCHW;
      break;
    case ACL_FORMAT_NHWC:
      output_format = ge::FORMAT_NHWC;
      break;
    case ACL_FORMAT_ND:
      output_format = ge::FORMAT_ND;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Failed to convert aclFormat(" << input_format
                           << ") to ge::Format.";
      break;
  }
  NNADAPTER_VLOG(5) << "geFormat: " << GEFormatToString(output_format);
  return output_format;
}

std::vector<int64_t> ConvertACLDimensions(aclmdlIODims input_dims) {
  std::vector<int64_t> output_dims(input_dims.dimCount);
  for (size_t i = 0; i < input_dims.dimCount; i++) {
    output_dims[i] = static_cast<int64_t>(input_dims.dims[i]);
  }
  return output_dims;
}

ge::DataType ConvertPrecision(NNAdapterOperandPrecisionCode input_precision) {
  ge::DataType output_precision = ge::DT_FLOAT;
  switch (input_precision) {
    case NNADAPTER_TENSOR_BOOL8:
      output_precision = ge::DT_BOOL;
      break;
    case NNADAPTER_TENSOR_INT8:
      output_precision = ge::DT_INT8;
      break;
    case NNADAPTER_TENSOR_INT16:
      output_precision = ge::DT_INT16;
      break;
    case NNADAPTER_TENSOR_INT32:
      output_precision = ge::DT_INT32;
      break;
    case NNADAPTER_TENSOR_INT64:
      output_precision = ge::DT_INT64;
      break;
    case NNADAPTER_TENSOR_UINT8:
      output_precision = ge::DT_UINT8;
      break;
    case NNADAPTER_TENSOR_QUANT_UINT8_ASYMM_PER_LAYER:
      output_precision = ge::DT_QUINT8;
      break;
    case NNADAPTER_TENSOR_UINT16:
      output_precision = ge::DT_UINT16;
      break;
    case NNADAPTER_TENSOR_UINT32:
      output_precision = ge::DT_UINT32;
      break;
    case NNADAPTER_TENSOR_UINT64:
      output_precision = ge::DT_UINT64;
      break;
    case NNADAPTER_TENSOR_FLOAT16:
      output_precision = ge::DT_FLOAT16;
      break;
    case NNADAPTER_TENSOR_FLOAT32:
      output_precision = ge::DT_FLOAT;
      break;
    case NNADAPTER_TENSOR_FLOAT64:
      output_precision = ge::DT_DOUBLE;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to ge::DataType !";
      break;
  }
  return output_precision;
}

ge::Format ConvertDataLayout(NNAdapterOperandLayoutCode input_layout) {
  ge::Format output_layout = ge::FORMAT_NCHW;
  switch (input_layout) {
    case NNADAPTER_NCHW:
      output_layout = ge::FORMAT_NCHW;
      break;
    case NNADAPTER_NHWC:
      output_layout = ge::FORMAT_NHWC;
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(input_layout) << ") to ge::Format !";
      break;
  }
  return output_layout;
}

std::vector<int64_t> ConvertDimensions(const int32_t* input_dimensions,
                                       uint32_t input_dimensions_count) {
  std::vector<int64_t> output_dimensions;
  for (uint32_t i = 0; i < input_dimensions_count; i++) {
    output_dimensions.push_back(input_dimensions[i]);
  }
  return output_dimensions;
}

std::vector<int64_t> ConvertDimensions(
    const std::vector<int32_t>& input_dimensions) {
  return ConvertDimensions(&input_dimensions[0], input_dimensions.size());
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
