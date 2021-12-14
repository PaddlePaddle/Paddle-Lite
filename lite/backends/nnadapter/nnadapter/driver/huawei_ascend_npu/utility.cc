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
#include <mutex>  // NOLINT
#include <regex>  // NOLINT
#include <utility>
#include "utility/debug.h"
#include "utility/string.h"

namespace nnadapter {
namespace huawei_ascend_npu {

static void FinalizeAscendCL() {
  NNADAPTER_VLOG(5) << "Finalize AscendCL.";
  // The following APIs can only be called once in one process
  aclFinalize();
}

void InitializeAscendCL() {
  static std::mutex mtx;
  static bool initialized = false;
  mtx.lock();
  if (!initialized) {
    NNADAPTER_VLOG(5) << "Initialize AscendCL.";
    // The following APIs can only be called once in one process
    aclInit(NULL);
    NNADAPTER_VLOG(5) << "Get AscendCL version.";
    GetAscendCANNVersion();
    // Register 'FinalizeAscendCL' to be called at normal process termination
    atexit(FinalizeAscendCL);
    initialized = true;
  }
  mtx.unlock();
}

static void FinalizeGraphBuilder() {
  NNADAPTER_VLOG(5) << "Finalize Graph Builder.";
  // The following APIs can only be called once in one process
  // TODO(hong19860320) fix the problem destruction order that the resource of
  // GE is released before the function is called.
  // ge::aclgrphBuildFinalize();
}

void InitializeGraphBuilder() {
  static std::mutex mtx;
  static bool initialized = false;
  mtx.lock();
  if (!initialized) {
    NNADAPTER_VLOG(5) << "Initialize Graph Builder.";
    // The following APIs can only be called once in one process
    ge::AscendString soc_version = GetAscendSocName();
    std::map<ge::AscendString, ge::AscendString> global_options;
    global_options.insert(
        std::make_pair(ge::ir_option::SOC_VERSION, soc_version));
    ge::aclgrphBuildInitialize(global_options);
    // Register 'FinalizeGraphBuilder' to be called at normal process
    // termination
    atexit(FinalizeGraphBuilder);
    initialized = true;
  }
  mtx.unlock();
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
    const std::vector<uint8_t>& model_buffer,
    int device_id,
    const std::string& profiling_file_path) {
  if (model_buffer.size() == 0) {
    NNADAPTER_LOG(ERROR) << "model_buffer size should not be 0!";
    return nullptr;
  }
  // Create a ACL model client to load the om model
  auto model_client =
      std::make_shared<AclModelClient>(device_id, profiling_file_path);
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
  // Should initialize the GE graph builder before model building
  InitializeGraphBuilder();
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
  options.insert(std::make_pair(ge::ir_option::OP_DEBUG_LEVEL, "0"));
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

ge::DataType ConvertACLDataTypeToGEDataType(aclDataType input_data_type) {
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

ge::Format ConvertACLFormatToGEFormat(aclFormat input_format) {
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

std::vector<int64_t> ConvertACLDimsToGEDims(
    const aclmdlIODims& input_dimensions) {
  std::vector<int64_t> output_dimensions(input_dimensions.dimCount);
  for (size_t i = 0; i < input_dimensions.dimCount; i++) {
    output_dimensions[i] = static_cast<int64_t>(input_dimensions.dims[i]);
  }
  return output_dimensions;
}

void ConvertACLDimsToGEDims(const aclmdlIODims& input_dimensions,
                            int32_t* output_dimensions,
                            uint32_t* output_dimensions_count) {
  for (size_t i = 0; i < input_dimensions.dimCount; i++) {
    output_dimensions[i] = static_cast<int32_t>(input_dimensions.dims[i]);
  }
}

ge::DataType ConvertToGEPrecision(
    NNAdapterOperandPrecisionCode precision_code) {
  switch (precision_code) {
    case NNADAPTER_BOOL8:
      return ge::DT_BOOL;
    case NNADAPTER_INT8:
    case NNADAPTER_QUANT_INT8_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT8_SYMM_PER_CHANNEL:
      return ge::DT_INT8;
    case NNADAPTER_INT16:
    case NNADAPTER_QUANT_INT16_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT16_SYMM_PER_CHANNEL:
      return ge::DT_INT16;
    case NNADAPTER_INT32:
    case NNADAPTER_QUANT_INT32_SYMM_PER_LAYER:
    case NNADAPTER_QUANT_INT32_SYMM_PER_CHANNEL:
      return ge::DT_INT32;
    case NNADAPTER_INT64:
      return ge::DT_INT64;
    case NNADAPTER_UINT8:
      return ge::DT_UINT8;
    case NNADAPTER_QUANT_UINT8_ASYMM_PER_LAYER:
      return ge::DT_QUINT8;
    case NNADAPTER_UINT16:
      return ge::DT_UINT16;
    case NNADAPTER_UINT32:
      return ge::DT_UINT32;
    case NNADAPTER_UINT64:
      return ge::DT_UINT64;
    case NNADAPTER_FLOAT16:
      return ge::DT_FLOAT16;
    case NNADAPTER_FLOAT32:
      return ge::DT_FLOAT;
    case NNADAPTER_FLOAT64:
      return ge::DT_DOUBLE;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(precision_code)
          << ") to ge::DataType !";
      break;
  }
  return ge::DT_FLOAT;
}

ge::Format ConvertToGEDataLayout(NNAdapterOperandLayoutCode layout_code) {
  switch (layout_code) {
    case NNADAPTER_NCHW:
      return ge::FORMAT_NCHW;
    case NNADAPTER_NHWC:
      return ge::FORMAT_NHWC;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand layout code("
          << OperandLayoutCodeToString(layout_code) << ") to ge::Format !";
      break;
  }
  return ge::FORMAT_NCHW;
}

std::vector<int64_t> ConvertToGEDimensions(const int32_t* input_dimensions,
                                           uint32_t input_dimensions_count) {
  std::vector<int64_t> output_dimensions;
  for (uint32_t i = 0; i < input_dimensions_count; i++) {
    output_dimensions.push_back(input_dimensions[i]);
  }
  return output_dimensions;
}

std::vector<int64_t> ConvertToGEDimensions(
    const std::vector<int32_t>& input_dimensions) {
  return ConvertToGEDimensions(&input_dimensions[0], input_dimensions.size());
}

std::string ConvertPadModeCodeToGEPadMode(int pad_mode_code) {
  switch (pad_mode_code) {
    case NNADAPTER_PAD_MODE_CONSTANT:
      return "constant";
    case NNADAPTER_PAD_MODE_REFLECT:
      return "reflect";
    case NNADAPTER_PAD_MODE_REPLICATE:
      return "replicate";
    case NNADAPTER_PAD_MODE_EDGE:
      return "edge";
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand pad mode code("
          << pad_mode_code << ") to pad mode !";
      break;
  }
  return "constant";
}

std::string ConvertInterpolateModeCodeToGEInterpolateMode(
    int interpolate_mode_code) {
  switch (interpolate_mode_code) {
    case NNADAPTER_INTERPOLATE_MODE_BILINEAR:
      return "bilinear";
    case NNADAPTER_INTERPOLATE_MODE_NEAREST:
      return "nearest";
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand interpolate mode code("
          << interpolate_mode_code << ") to interpolate mode !";
      break;
  }
  return "bilinear";
}

std::string GetRealPath(const char* path) {
  if (path == nullptr) {
    NNADAPTER_LOG(FATAL) << "Input path is nullptr";
    return nullptr;
  }

  char real_path[4096] = {0};
  if (strlen(path) >= 4096 || realpath(path, real_path) == nullptr) {
    NNADAPTER_LOG(FATAL) << "Get real path failed, path[" << path << "]";
    return nullptr;
  }
  return std::string(real_path);
}

inline void GetAscendCANNVersion() {
  std::string ascend_cann_path;
  std::string ld_library_paths = GetStringFromEnv(LD_LIBRARY_PATH);
  std::vector<std::string> paths =
      string_split<std::string>(ld_library_paths, ":");
  for (auto path : paths) {
    if (path.find("ascend-toolkit") != std::string::npos) {
      ascend_cann_path = path;
      break;
    } else {
      ascend_cann_path = "/usr/local/Ascend/ascend-toolkit/latest";
    }
  }
  auto ascend_cann_real_path = GetRealPath(ascend_cann_path.c_str());
  std::vector<std::string> path_split =
      string_split<std::string>(ascend_cann_real_path, "/");
  std::string cann_version;
  for (size_t i = 0; i < path_split.size(); i++) {
    if (path_split[i] == "ascend-toolkit") {
      if (i <= path_split.size() - 1) {
        cann_version = path_split[i + 1];
      }
      break;
    }
  }
  auto cann_version_split = string_split<std::string>(cann_version, ".");
  if (cann_version_split.size() == 3) {
    major_version = atoi(cann_version_split[0].c_str());
    minor_version = atoi(cann_version_split[1].c_str());
    patch_version = atoi(cann_version_split[2].c_str());
  } else if (cann_version_split.size() == 4) {
    major_version = atoi(cann_version_split[0].c_str());
    minor_version = atoi(cann_version_split[1].c_str());
    patch_version = atoi(cann_version_split[2].c_str());
    std::regex reg("(\\d+)");
    std::smatch matchResult;
    if (std::regex_search(cann_version_split[3], matchResult, reg)) {
      if (matchResult.size() >= 1) {
        bugfix_version = atoi(matchResult[0].str().c_str());
      } else {
        bugfix_version = 0;
      }
    }
  } else {
    NNADAPTER_LOG(FATAL) << "Can't get cann version";
  }
}

inline std::string ComposeCANNVersion() {
  if (bugfix_version == -1)
    return string_format(
        "%d.%d.%d", major_version, minor_version, patch_version);
  else
    return string_format("%d.%d.%d.alpha00%d",
                         major_version,
                         minor_version,
                         patch_version,
                         bugfix_version);
}

inline std::string SplitCANNVersion() {
  if (bugfix_version == -1) {
    return string_format(
        "%d.%d.%d", major_version, minor_version, patch_version);
  } else {
    int major = NNADAPTE_HUAWEI_ASCEND_NPU_CANN_VERSION / 1000;
    int minor = NNADAPTE_HUAWEI_ASCEND_NPU_CANN_VERSION / 100;
    int patch = NNADAPTE_HUAWEI_ASCEND_NPU_CANN_VERSION / 10;
    int bugfix = NNADAPTE_HUAWEI_ASCEND_NPU_CANN_VERSION / 1;
    return string_format("%d.%d.%d.alpha00%d", major, minor, patch, bugfix);
  }
}

inline ge::AscendString GetAscendSocName() {
  ge::AscendString soc_version = "Ascend310";
#if NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MIN_VERSION(5, 0, 2, 1)
  if (major_version >= 5 && minor_version >= 0 && patch_version >= 2) {
    const char* soc_name = aclrtGetSocName();
    if (soc_name != nullptr) {
      soc_version = ge::AscendString(soc_name);
    } else {
      NNADAPTER_LOG(WARNING)
          << "Get Ascend soc name failed. Ascend310 is used by default";
    }
  } else {
    NNADAPTER_LOG(FATAL) << "The Ascend CANN version you actually installed("
                         << ComposeCANNVersion() << ")"
                         << "is lower than the minimum version you specified("
                         << SplitCANNVersion() << ")";
  }
#endif
  return soc_version;
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
