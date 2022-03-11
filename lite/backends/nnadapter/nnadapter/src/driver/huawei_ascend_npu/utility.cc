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
    int major_version = 0, minor_version = 0, patch_version = 0;
    GetAscendCANNVersion(&major_version, &minor_version, &patch_version);
    auto current_version =
        string_format("%d.%d.%d",
                      NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MAJOR_VERSION,
                      NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MINOR_VERSION,
                      NNADAPTER_HUAWEI_ASCEND_NPU_CANN_PATCH_VERSION);
    auto build_version =
        string_format("%d.%d.%d", major_version, minor_version, patch_version);
    NNADAPTER_VLOG(5) << "The current library is compiled based on CANN "
                      << current_version;
    NNADAPTER_VLOG(5) << "The CANN version of the current environment is "
                      << build_version;
    if (major_version != NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MAJOR_VERSION &&
        minor_version != NNADAPTER_HUAWEI_ASCEND_NPU_CANN_MINOR_VERSION &&
        patch_version != NNADAPTER_HUAWEI_ASCEND_NPU_CANN_PATCH_VERSION) {
      NNADAPTER_LOG(WARNING)
          << "CANN version mismatch. The build version is " << build_version
          << ", but the current environment version is " << current_version
          << ".";
    }
    NNADAPTER_VLOG(5) << "Initialize AscendCL.";
    // The following APIs can only be called once in one process
    aclInit(NULL);
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

void InitializeGraphBuilder(AscendConfigParams* config_params) {
  static std::mutex mtx;
  static bool initialized = false;
  mtx.lock();
  if (!initialized) {
    NNADAPTER_VLOG(5) << "Initialize Graph Builder.";
    // The following APIs can only be called once in one process
    ge::AscendString soc_version = GetAscendSocName();
    NNADAPTER_VLOG(5) << "Initialize the Graph Builder based on SoC name: "
                      << soc_version.GetString();
    std::map<ge::AscendString, ge::AscendString> global_options;
    global_options.insert(
        std::make_pair(ge::ir_option::SOC_VERSION, soc_version));
#if NNADAPTER_HUAWEI_ASCEND_NPU_CANN_VERSION_GREATER_THAN(5, 0, 2)
    global_options.insert(std::make_pair(ge::ir_option::OP_DEBUG_LEVEL, "0"));
    global_options.insert(std::make_pair(ge::ir_option::DEBUG_DIR, "/tmp/"));
    auto precision_mode = config_params->precision_mode;
    if (!precision_mode.empty()) {
      global_options.insert(
          std::make_pair(ge::ir_option::PRECISION_MODE,
                         ge::AscendString(precision_mode.c_str())));
      if (precision_mode == "allow_mix_precision" &&
          !config_params->modify_mixlist_path.empty()) {
        global_options.insert(std::make_pair(
            ge::ir_option::MODIFY_MIXLIST,
            ge::AscendString(config_params->modify_mixlist_path.c_str())));
      }
    }
    if (!config_params->op_select_impl_mode.empty() &&
        !config_params->op_type_list_for_impl_mode.empty()) {
      global_options.insert(
          std::make_pair(ge::ir_option::OP_SELECT_IMPL_MODE,
                         config_params->op_select_impl_mode.c_str()));
      global_options.insert(
          std::make_pair(ge::ir_option::OPTYPELIST_FOR_IMPLMODE,
                         config_params->op_type_list_for_impl_mode.c_str()));
    }
    if (!config_params->enable_compress_weight.empty()) {
      global_options.insert(
          std::make_pair(ge::ir_option::ENABLE_COMPRESS_WEIGHT,
                         config_params->enable_compress_weight.c_str()));
    }
    if (!config_params->auto_tune_mode.empty()) {
      global_options.insert(
          std::make_pair(ge::ir_option::AUTO_TUNE_MODE,
                         config_params->auto_tune_mode.c_str()));
    }
#endif
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
    AscendConfigParams* config_params) {
  if (model_buffer.size() == 0) {
    NNADAPTER_LOG(ERROR) << "model_buffer size should not be 0!";
    return nullptr;
  }
  // Create a ACL model client to load the om model
  auto model_client =
      std::make_shared<AclModelClient>(device_id, config_params);
  // Load model from memory
  if (model_client->LoadModel(
          reinterpret_cast<const void*>(model_buffer.data()),
          model_buffer.size(),
          config_params)) {
    return model_client;
  }
  return nullptr;
}

bool BuildOMModelToBuffer(
    std::vector<ge::Operator>& input_operators,   // NOLINT
    std::vector<ge::Operator>& output_operators,  // NOLINT
    std::vector<uint8_t>* model_buffer,
    const std::vector<std::string>& dynamic_shape_info,
    const std::string& optional_shape_str,
    const DynamicShapeMode dynamic_shape_mode,
    AscendConfigParams* config_params) {
  // Should initialize the GE graph builder before model building
  InitializeGraphBuilder(config_params);
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

  std::string input_shape_info;
  for (size_t i = 0; i < dynamic_shape_info.size(); i++) {
    if (!dynamic_shape_info[i].empty()) {
      ge::AscendString name;
      input_operators[i].GetName(name);
      input_shape_info +=
          std::string(name.GetString()) + ":" + dynamic_shape_info[i] + ";";
    }
  }
  NNADAPTER_CHECK(!input_shape_info.empty());
  input_shape_info.pop_back();

  if (!optional_shape_str.empty()) {
    if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_BATCH_SIZE) {
      options.insert(std::make_pair(ge::ir_option::DYNAMIC_BATCH_SIZE,
                                    optional_shape_str.data()));
      options.insert(std::make_pair(ge::ir_option::INPUT_FORMAT, "NCHW"));
    } else if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH) {
      options.insert(std::make_pair(ge::ir_option::DYNAMIC_IMAGE_SIZE,
                                    optional_shape_str.data()));
      options.insert(std::make_pair(ge::ir_option::INPUT_FORMAT, "NCHW"));
    } else if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_N_DIMS) {
      options.insert(std::make_pair(ge::ir_option::DYNAMIC_DIMS,
                                    optional_shape_str.data()));
      options.insert(std::make_pair(ge::ir_option::INPUT_FORMAT, "ND"));
    }
  } else {
    options.insert(std::make_pair(ge::ir_option::INPUT_FORMAT, "NCHW"));
  }

  if (dynamic_shape_mode == DYNAMIC_SHAPE_MODE_SHAPE_RANGE) {
#if NNADAPTER_HUAWEI_ASCEND_NPU_CANN_VERSION_GREATER_THAN(5, 1, 1)
    options.insert(std::make_pair(ge::ir_option::INPUT_SHAPE_RANGE,
                                  input_shape_info.data()));
#else
    NNADAPTER_LOG(FATAL)
        << "The dynamic shape range feature is only supported in CANN 5.1.1 "
           "and above."
        << "If you want to use, please upgrade CANN version and recompile the "
           "library.";
#endif
  } else {
    options.insert(
        std::make_pair(ge::ir_option::INPUT_SHAPE, input_shape_info.data()));
  }
  ATC_CALL(aclgrphBuildModel(ir_graph, options, om_buffer));
  // For debug: save ascend offline model to local.
  if (!config_params->dump_model_path.empty()) {
    ATC_CALL(aclgrphSaveModel(
        std::string(config_params->dump_model_path + "ir_graph_model").c_str(),
        om_buffer));
  }
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
    if (input_dimensions[i] == NNADAPTER_UNKNOWN) {
      output_dimensions.push_back(-1);
    } else {
      output_dimensions.push_back(input_dimensions[i]);
    }
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

bool GetAscendCANNVersion(int* major, int* minor, int* patch) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  static bool initialized = false;
  static int major_version = 0;
  static int minor_version = 0;
  static int patch_version = 0;
  if (!initialized) {
    initialized = true;
    std::string ld_library_path = GetStringFromEnv("LD_LIBRARY_PATH");
    // Split the value of LD_LIBRARY_PATH by ":", and obtain the root directory
    // of the Ascend CANN toolkit
    std::vector<std::string> paths =
        string_split<std::string>(ld_library_path, ":");
    paths.push_back("/usr/local/Ascend/ascend-toolkit/latest");
    for (auto path : paths) {
      if (path.find("Ascend/ascend-toolkit") == std::string::npos &&
          path.find("Ascend/nnrt") == std::string::npos)
        continue;
      auto ascend_cann_path = GetRealPath(path.c_str());
      // Check if the file path is valid
      if (ascend_cann_path.empty()) continue;
      // Split ascend_cann_path string by "/"
      auto tokens = string_split<std::string>(ascend_cann_path, "/");
      std::string ascend_cann_version;
      for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i] == "ascend-toolkit" || tokens[i] == "nnrt") {
          if (i < tokens.size() - 1) {
            ascend_cann_version = tokens[i + 1];
            break;
          }
        }
      }
      if (ascend_cann_version.empty()) continue;
      // Split ascend_cann_version string by "."
      tokens = string_split<std::string>(ascend_cann_version, ".");
      if (tokens.size() == 3 || tokens.size() == 4) {
        major_version = atoi(tokens[0].c_str());
        minor_version = atoi(tokens[1].c_str());
        patch_version = atoi(tokens[2].c_str());
        return true;
      }
    }
    NNADAPTER_LOG(FATAL) << "Unable to find the Ascend CANN installation path! "
                            "Please install the Ascend CANN and add the root "
                            "directory into LD_LIBRARY_PATH.";
    return false;
  }
  *major = major_version;
  *minor = minor_version;
  *patch = patch_version;
  return true;
}

ge::AscendString GetAscendSocName() {
  ge::AscendString soc_version = "Ascend310";
#if NNADAPTER_HUAWEI_ASCEND_NPU_CANN_VERSION_GREATER_THAN(5, 0, 2)
  const char* soc_name = aclrtGetSocName();
  if (soc_name) {
    soc_version = ge::AscendString(soc_name);
  } else {
    NNADAPTER_LOG(WARNING) << "Failed to call aclrtGetSocName to obtain the "
                              "SoC name, so Ascend 310 is used by default.";
  }
#else
  NNADAPTER_LOG(WARNING) << "Since the current library is compiled based on "
                            "CANN versions below 5.0.2, aclrtGetSocName "
                            "cannot be called to obtain the SoC name of the "
                            "current device, so Ascend 310 is used by default. "
                            "If you want to use ascend 710, please recompile "
                            "the library based on CANN 5.0.2 and above.";
#endif
  return soc_version;
}

std::string ShapeToString(const std::vector<int32_t>& shape) {
  std::string shape_str;
  for (size_t i = 0; i < shape.size(); i++) {
    if (!shape_str.empty()) {
      shape_str += ",";
    }
    shape_str += std::to_string(shape[i]);
  }
  return shape_str;
}

std::string MergeOptionalShapInfo(
    const std::vector<std::string>& optional_shape_info,
    const DynamicShapeMode dynamic_shape_mode) {
  std::string merged_shape_str;
  switch (dynamic_shape_mode) {
    case DYNAMIC_SHAPE_MODE_NONE:
    case DYNAMIC_SHAPE_MODE_SHAPE_RANGE:
      break;
    case DYNAMIC_SHAPE_MODE_BATCH_SIZE: {
      for (auto shape_info : optional_shape_info) {
        merged_shape_str += shape_info + ",";
      }
      merged_shape_str.pop_back();
    } break;
    case DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH:
    case DYNAMIC_SHAPE_MODE_N_DIMS: {
      for (auto shape_info : optional_shape_info) {
        merged_shape_str += shape_info + ";";
      }
      merged_shape_str.pop_back();
    } break;
    default:
      NNADAPTER_LOG(FATAL) << "Unsupported dynamic_shape_mode: "
                           << dynamic_shape_mode;
      break;
  }
  return merged_shape_str;
}

static void UpdateDynamicShapeMode(
    const NNAdapterOperandDimensionType& dimensions,
    DynamicShapeMode* dynamic_shape_mode) {
  bool is_nchw = dimensions.count == 4;
  bool b_unk = dimensions.data[0] == NNADAPTER_UNKNOWN;
  bool c_unk = dimensions.data[1] == NNADAPTER_UNKNOWN;
  bool h_unk = dimensions.data[2] == NNADAPTER_UNKNOWN;
  bool w_unk = dimensions.data[3] == NNADAPTER_UNKNOWN;
  if (is_nchw && b_unk && !c_unk && !h_unk && !w_unk) {
    if (*dynamic_shape_mode == DYNAMIC_SHAPE_MODE_NONE) {
      *dynamic_shape_mode = DYNAMIC_SHAPE_MODE_BATCH_SIZE;
    }
    if (*dynamic_shape_mode != DYNAMIC_SHAPE_MODE_BATCH_SIZE) {
      *dynamic_shape_mode = DYNAMIC_SHAPE_MODE_N_DIMS;
    }
  } else if (is_nchw && !b_unk && !c_unk && (h_unk || w_unk)) {
    if (*dynamic_shape_mode == DYNAMIC_SHAPE_MODE_NONE) {
      *dynamic_shape_mode = DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH;
    } else {
      // only support one input has dynamic h&w
      *dynamic_shape_mode = DYNAMIC_SHAPE_MODE_N_DIMS;
    }
  } else {
    *dynamic_shape_mode = DYNAMIC_SHAPE_MODE_N_DIMS;
  }
}

void GetDynamicShapeInfo(const std::vector<NNAdapterOperandType>& input_types,
                         std::vector<std::string>* dynamic_shape_info,
                         std::string* optional_shape_str,
                         DynamicShapeMode* dynamic_shape_mode) {
  // Get dynamic_shape_mode from all inputs. Rules are as follows:
  // 1. If all shapes are const, dynamic_shape_mode is DYNAMIC_SHAPE_MODE_NONE.
  // 2. If only batch of inputs is unknown, dynamic_shape_mode is
  // DYNAMIC_SHAPE_MODE_BATCH_SIZE.
  // 3. If only one 4-D input has dynamic height or weight, dynamic_shape_mode
  // is DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH.
  // 4. Others belong to DYNAMIC_SHAPE_MODE_N_DIMS.
  if (*dynamic_shape_mode != DYNAMIC_SHAPE_MODE_SHAPE_RANGE) {
    for (auto& input_type : input_types) {
      if (!IsDynamicShapeOperandType(input_type)) continue;
      UpdateDynamicShapeMode(input_type.dimensions, dynamic_shape_mode);
      if (*dynamic_shape_mode == DYNAMIC_SHAPE_MODE_N_DIMS) break;
    }
  }
  // Generate dynamic_shape_info according to dynamic_shape_mode.
  std::vector<std::string> optional_shape;
  for (auto& input_type : input_types) {
    auto dimensions = input_type.dimensions;
    std::vector<int32_t> shape;
    for (size_t i = 0; i < dimensions.count; i++) {
      if (dimensions.data[i] == NNADAPTER_UNKNOWN) {
        shape.push_back(-1);
      } else {
        shape.push_back(dimensions.data[i]);
      }
    }
    if (!IsDynamicShapeOperandType(input_type)) {
      dynamic_shape_info->push_back(ShapeToString(shape));
      continue;
    }
    // Fill optional_shape if input_type has dynamic shapes.
    NNADAPTER_CHECK_GT(dimensions.dynamic_count, 0U);
    optional_shape.resize(dimensions.dynamic_count);
    switch (*dynamic_shape_mode) {
      case DYNAMIC_SHAPE_MODE_BATCH_SIZE: {
        dynamic_shape_info->push_back(ShapeToString(shape));
        for (size_t i = 0; i < optional_shape.size(); i++) {
          auto& optional_batch_str = optional_shape.at(i);
          auto dynamic_batch_str =
              std::to_string(dimensions.dynamic_data[i][0]);
          if (optional_batch_str.empty()) {
            optional_batch_str = dynamic_batch_str;
          }
          NNADAPTER_CHECK_EQ(optional_batch_str, dynamic_batch_str);
        }
      } break;
      case DYNAMIC_SHAPE_MODE_HEIGHT_WIDTH: {
        NNADAPTER_CHECK_EQ(shape.size(), 4UL);
        shape[2] = -1;
        shape[3] = -1;
        dynamic_shape_info->push_back(ShapeToString(shape));
        for (size_t i = 0; i < optional_shape.size(); i++) {
          auto& optional_hw_str = optional_shape.at(i);
          NNADAPTER_CHECK(optional_hw_str.empty());
          optional_hw_str = std::to_string(dimensions.dynamic_data[i][2]) +
                            "," + std::to_string(dimensions.dynamic_data[i][3]);
        }
      } break;
      case DYNAMIC_SHAPE_MODE_N_DIMS: {
        dynamic_shape_info->push_back(ShapeToString(shape));
        for (size_t i = 0; i < optional_shape.size(); i++) {
          auto& optional_ndims_str = optional_shape.at(i);
          for (uint32_t j = 0; j < dimensions.count; j++) {
            if (dimensions.data[j] != NNADAPTER_UNKNOWN) continue;
            if (!optional_ndims_str.empty()) {
              optional_ndims_str += ",";
            }
            optional_ndims_str += std::to_string(dimensions.dynamic_data[i][j]);
          }
        }
      } break;
      case DYNAMIC_SHAPE_MODE_SHAPE_RANGE: {
        std::string shape_range_str;
        if (optional_shape.size() == 1) {
          for (size_t i = 0; i < dimensions.count; i++) {
            shape_range_str +=
                std::to_string(dimensions.dynamic_data[0][i]) + ",";
          }
        } else if (optional_shape.size() == 2) {
          for (size_t i = 0; i < dimensions.count; i++) {
            if (dimensions.dynamic_data[0][i] !=
                dimensions.dynamic_data[1][i]) {
              NNADAPTER_CHECK_LT(dimensions.dynamic_data[0][i],
                                 dimensions.dynamic_data[1][i])
                  << "The value of the maximum gear shape should be greater "
                     "than the value of the corresponding minimum gear shape.";
              shape_range_str +=
                  std::to_string(dimensions.dynamic_data[0][i]) + "~" +
                  std::to_string(dimensions.dynamic_data[1][i]) + ",";
            } else {
              shape_range_str +=
                  std::to_string(dimensions.dynamic_data[0][i]) + ",";
            }
          }
        } else {
          NNADAPTER_LOG(FATAL)
              << "DYNAMIC_SHAPE_MODE_SHAPE_RANGE only supports dynamic "
                 "dimension count equal to 1 or 2, but the given dynamic "
                 "dimension count is "
              << optional_shape.size();
        }
        shape_range_str.pop_back();
        shape_range_str = "[" + shape_range_str + "]";
        dynamic_shape_info->push_back(shape_range_str);
      } break;
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported dynamic_shape_mode: "
                             << dynamic_shape_mode;
        break;
    }
  }
  // Generate optional_shape_str.
  *optional_shape_str =
      MergeOptionalShapInfo(optional_shape, *dynamic_shape_mode);
}

}  // namespace huawei_ascend_npu
}  // namespace nnadapter
