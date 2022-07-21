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

#include "driver/qualcomm_qnn/engine.h"
#include <utility>
#include "driver/qualcomm_qnn/optimizer/convert_datalayout_nchw_to_nhwc.h"
#include "driver/qualcomm_qnn/optimizer/fuse_yolobox3d_nms3.h"
#include "driver/qualcomm_qnn/optimizer/restrict_input_output_quant_params.h"
#include "driver/qualcomm_qnn/optimizer/unpack_op_fusion.h"
#include "optimizer/convert_quantization_symm_to_asymm.h"
#include "optimizer/fuse_matmul_add_into_fully_connected.h"

namespace nnadapter {
namespace qualcomm_qnn {

void* Program::lib_backend_handle_{nullptr};
QNN_INTERFACE_VER_TYPE Program::qnn_interface_{nullptr};
const QnnBackend_Config_t* Program::qnn_backend_configs_{nullptr};

static core::Argument* FindArgumentByIndex(core::Argument* arguments,
                                           int index,
                                           uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    if (arguments[i].index == index) {
      return &arguments[i];
    }
  }
  return static_cast<core::Argument*>(nullptr);
}

Context::Context(void* device, const char* properties) : device_(device) {
  // Extract the runtime parameters from the context properties
  NNADAPTER_VLOG(1) << "properties: " << std::string(properties);
  auto key_values = GetKeyValues(properties);
  // Device type
  std::string device_type = key_values.count(QUALCOMM_QNN_DEVICE)
                                ? key_values[QUALCOMM_QNN_DEVICE]
                                : GetStringFromEnv(QUALCOMM_QNN_DEVICE);
  NNADAPTER_CHECK(!device_type.empty());
  std::map<std::string, DeviceType> device_map{
      {"cpu", kCPU}, {"gpu", kGPU}, {"htp", kHTP}};
  NNADAPTER_CHECK(device_map.count(device_type)) << "Not support devive_type: "
                                                 << device_type;
  device_type_ = device_map[device_type];
  // Runtime lib
  std::map<std::string, std::string> runtime_lib_map{{"cpu", "libQnnCpu.so"},
                                                     {"htp", "libQnnHtp.so"}};
  NNADAPTER_CHECK(runtime_lib_map.count(device_type))
      << "Not support devive_type: " << device_type;
  runtime_lib_ = runtime_lib_map[device_type];
  // Log level
  std::string log_level = key_values.count(QUALCOMM_QNN_LOG_LEVEL)
                              ? key_values[QUALCOMM_QNN_LOG_LEVEL]
                              : GetStringFromEnv(QUALCOMM_QNN_LOG_LEVEL);
  std::map<std::string, QnnLog_Level_t> log_level_map{
      {"error", QNN_LOG_LEVEL_ERROR},
      {"warn", QNN_LOG_LEVEL_WARN},
      {"info", QNN_LOG_LEVEL_INFO},
      {"verbose", QNN_LOG_LEVEL_VERBOSE},
      {"debug", QNN_LOG_LEVEL_DEBUG}};
  if (!log_level.empty()) {
    NNADAPTER_CHECK(log_level_map.count(log_level)) << "Not support log_level: "
                                                    << log_level;
    log_level_ = log_level_map[log_level];
  }
}

Program::Program(Context* context) : context_(context) {
  if (!lib_backend_handle_) {
    lib_backend_handle_ =
        dlopen(context_->RuntimeLib().c_str(), RTLD_NOW | RTLD_LOCAL);
    NNADAPTER_CHECK(lib_backend_handle_) << dlerror();
    qnn_interface_ = GetQnnInterface(lib_backend_handle_);
    QNN_CHECK(
        qnn_interface_.logInitialize(LogStdoutCallback, context_->LogLevel()));
    // Init backend
    QNN_CHECK(qnn_interface_.backendInitialize(&qnn_backend_configs_));
  }
  // Create context
  QNN_CHECK(qnn_interface_.contextCreate(&qnn_context_configs_, &qnn_context_));
}

Program::~Program() { Clear(); }

void Program::Clear() {
  tensors_.clear();
  input_types_.clear();
  output_types_.clear();
  input_tensors_.clear();
  output_tensors_.clear();
  input_dims_.clear();
  output_dims_.clear();
  input_tensor_ids_.clear();
  output_tensor_ids_.clear();
}

int Program::Build(core::Model* model, core::Cache* cache) {
  Clear();
  if (cache->buffer.empty()) {
    NNADAPTER_CHECK_EQ(BuildFromModel(model), NNADAPTER_NO_ERROR);
    SerializeToCache(&cache->buffer);
  } else {
    DeserializeFromCache(&cache->buffer);
    NNADAPTER_CHECK_EQ(BuildFromCache(cache), NNADAPTER_NO_ERROR);
  }
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromModel(core::Model* model) {
  // Optimzie model
  NNADAPTER_VLOG(5) << "Origin model:" << std::endl << Visualize(model);
  FuseMatMulAddIntoFullyConnected(model);
  FuseYoloBox3dNms(model);
  UnpackOpFusion(model);
  ConvertQuantizationSymmToAsymm(model);
  RestrictInputOutputQuantParams(model);
  ConvertDataLayoutNCHWToNHWC(model);
  NNADAPTER_VLOG(5) << "Optimized model:" << std::endl << Visualize(model);
  // Create graph
  graph_name_ = "subgraph_" + std::to_string(reinterpret_cast<uint64_t>(model));
  QNN_CHECK(qnn_interface_.graphCreate(qnn_context_,
                                       graph_name_.c_str(),
                                       &qnn_graph_config_,
                                       &qnn_graph_handle_));
  Converter converter(
      qnn_interface_, &qnn_graph_handle_, &tensors_, context_->Device());
  NNADAPTER_CHECK_EQ(converter.Apply(model), NNADAPTER_NO_ERROR);
  QNN_CHECK(qnn_interface_.graphFinalize(qnn_graph_handle_, nullptr, nullptr));
  for (auto input_operand : model->input_operands) {
    input_types_.push_back(input_operand->type);
    input_tensor_ids_.push_back(tensors_.at(input_operand).back().id);
  }
  for (auto output_operand : model->output_operands) {
    output_types_.push_back(output_operand->type);
    output_tensor_ids_.push_back(tensors_.at(output_operand).back().id);
  }
  return NNADAPTER_NO_ERROR;
}

int Program::BuildFromCache(core::Cache* cache) {
  QNN_CHECK(qnn_interface_.graphRetrieve(
      qnn_context_, graph_name_.c_str(), &qnn_graph_handle_));
  input_types_ = cache->input_types;
  output_types_ = cache->output_types;
  return NNADAPTER_NO_ERROR;
}

int Program::CheckInputsAndOutputs(uint32_t input_count,
                                   core::Argument* input_arguments,
                                   uint32_t output_count,
                                   core::Argument* output_arguments) {
  return NNADAPTER_NO_ERROR;
}

int Program::Execute(uint32_t input_count,
                     core::Argument* input_arguments,
                     uint32_t output_count,
                     core::Argument* output_arguments) {
  int ret = CheckInputsAndOutputs(
      input_count, input_arguments, output_count, output_arguments);
  if (ret != NNADAPTER_NO_ERROR) return ret;
  // Prepare input
  if (input_tensors_.empty()) {
    input_tensors_.resize(input_count);
    input_dims_.resize(input_count);
    for (size_t i = 0; i < input_types_.size(); i++) {
      InitInputTensor(i);
    }
  }
  for (uint32_t i = 0; i < input_count; i++) {
    auto arg = FindArgumentByIndex(input_arguments, i, input_count);
    NNADAPTER_CHECK(arg) << "Input argument " << i << " does not exist!";
    auto& type = input_types_.at(i);
    auto buffer = arg->access(arg->memory, &type, nullptr);
    auto length = GetOperandTypeBufferLength(type);
    if (IsUInt8AsymmPerLayerQuantType(type.precision)) {
      Symm2AsymmData(reinterpret_cast<const int8_t*>(buffer),
                     length,
                     type.asymm_per_layer_params.zero_point,
                     reinterpret_cast<uint8_t*>(buffer));
    }
    input_tensors_.at(i).clientBuf.data = buffer;
    input_tensors_.at(i).clientBuf.dataSize = length;
  }
  // Prepare output
  if (output_tensors_.empty()) {
    output_tensors_.resize(output_count);
    output_dims_.resize(output_count);
    for (size_t i = 0; i < output_types_.size(); i++) {
      InitOutputTensor(i);
    }
  }
  std::vector<std::pair<void*, size_t>> output_buffers(output_count);
  for (uint32_t i = 0; i < output_count; i++) {
    auto arg = FindArgumentByIndex(output_arguments, i, output_count);
    NNADAPTER_CHECK(arg) << "Output argument " << i << " does not exist!";
    auto& type = output_types_.at(i);
    auto buffer = arg->access(arg->memory, &type, nullptr);
    auto length = GetOperandTypeBufferLength(type);
    output_tensors_.at(i).clientBuf.data = buffer;
    output_tensors_.at(i).clientBuf.dataSize = length;
    output_buffers[i].first = buffer;
    output_buffers[i].second = length;
    NNADAPTER_LOG(INFO) << "before: output_tensors_.at(i).clientBuf.dataSize"
                        << output_tensors_.at(i).clientBuf.dataSize;
    NNADAPTER_LOG(INFO) << "before: output_tensors_.at(i).clientBuf.data"
                        << reinterpret_cast<int32_t*>(
                               output_tensors_.at(i).clientBuf.data)[0];
    NNADAPTER_LOG(INFO) << "before: output_tensors_.at(i).currentDimensions[0]:"
                        << output_tensors_.at(i).currentDimensions[0];
    NNADAPTER_LOG(INFO) << "before: output_tensors_.at(i).currentDimensions[0]:"
                        << output_tensors_.at(i).currentDimensions[1];
    NNADAPTER_LOG(INFO) << "before: output_tensors_.at(i).currentDimensions[0]:"
                        << output_tensors_.at(i).currentDimensions[2];
    NNADAPTER_LOG(INFO) << "before: output_tensors_.at(i).currentDimensions[0]:"
                        << output_tensors_.at(i).currentDimensions[3];
  }
  // Execute graph
  QNN_CHECK(qnn_interface_.graphExecute(qnn_graph_handle_,
                                        input_tensors_.data(),
                                        input_tensors_.size(),
                                        output_tensors_.data(),
                                        output_tensors_.size(),
                                        nullptr,
                                        nullptr));
  for (uint32_t i = 0; i < output_count; i++) {
    NNADAPTER_LOG(INFO) << "after: output_tensors_.at(i).clientBuf.dataSize"
                        << output_tensors_.at(i).clientBuf.dataSize;
    NNADAPTER_LOG(INFO) << "after: output_tensors_.at(i).clientBuf.data"
                        << reinterpret_cast<int32_t*>(
                               output_tensors_.at(i).clientBuf.data)[0];
    NNADAPTER_LOG(INFO) << "after: output_tensors_.at(i).currentDimensions[0]:"
                        << output_tensors_.at(i).currentDimensions[0];
    NNADAPTER_LOG(INFO) << "after: output_tensors_.at(i).currentDimensions[0]:"
                        << output_tensors_.at(i).currentDimensions[1];
    NNADAPTER_LOG(INFO) << "after: output_tensors_.at(i).currentDimensions[0]:"
                        << output_tensors_.at(i).currentDimensions[2];
    NNADAPTER_LOG(INFO) << "after: output_tensors_.at(i).currentDimensions[0]:"
                        << output_tensors_.at(i).currentDimensions[3];
    auto& type = output_types_[i];
    auto buffer = output_buffers[i].first;
    auto length = output_buffers[i].second;
    if (IsUInt8AsymmPerLayerQuantType(type.precision)) {
      Asymm2SymmData(reinterpret_cast<const uint8_t*>(buffer),
                     length,
                     type.asymm_per_layer_params.zero_point,
                     reinterpret_cast<int8_t*>(buffer));
    }
  }
  return NNADAPTER_NO_ERROR;
}

void Program::SerializeToCache(std::vector<uint8_t>* buffer) {
  // Calculate total size
  if (context_->Device() == kCPU) return;
  Qnn_ContextBinarySize_t binary_size{0};
  QNN_CHECK(qnn_interface_.contextGetBinarySize(qnn_context_, &binary_size));
  NNADAPTER_CHECK_GT(binary_size, 0);
  size_t graph_name_size = graph_name_.size();
  size_t input_tensor_ids_size = input_tensor_ids_.size() * sizeof(uint32_t);
  size_t output_tensor_ids_size = output_tensor_ids_.size() * sizeof(uint32_t);
  size_t size_all = sizeof(graph_name_size) + graph_name_size +
                    sizeof(input_tensor_ids_size) + input_tensor_ids_size +
                    sizeof(output_tensor_ids_size) + output_tensor_ids_size +
                    static_cast<size_t>(binary_size);
  buffer->resize(size_all);
  uint8_t* buffer_data = buffer->data();
  // Serialize graph_name_
  std::memcpy(buffer_data, &graph_name_size, sizeof(graph_name_size));
  buffer_data += sizeof(graph_name_size);
  std::memcpy(buffer_data, graph_name_.data(), graph_name_.size());
  buffer_data += graph_name_.size();
  // Serialize input_tensor_ids_
  std::memcpy(
      buffer_data, &input_tensor_ids_size, sizeof(input_tensor_ids_size));
  buffer_data += sizeof(input_tensor_ids_size);
  std::memcpy(buffer_data, input_tensor_ids_.data(), input_tensor_ids_size);
  buffer_data += input_tensor_ids_size;
  // Serialize output_tensor_ids_
  std::memcpy(
      buffer_data, &output_tensor_ids_size, sizeof(output_tensor_ids_size));
  buffer_data += sizeof(output_tensor_ids_size);
  std::memcpy(buffer_data, output_tensor_ids_.data(), output_tensor_ids_size);
  buffer_data += output_tensor_ids_size;
  // Serialize qnn_context
  Qnn_ContextBinarySize_t write_binary_size = 0;
  QNN_CHECK(qnn_interface_.contextGetBinary(
      qnn_context_, buffer_data, binary_size, &write_binary_size));
  NNADAPTER_CHECK_EQ(binary_size, write_binary_size);
}

void Program::DeserializeFromCache(std::vector<uint8_t>* buffer) {
  NNADAPTER_CHECK(context_->Device() != kCPU)
      << "Qualcomm qnn cpu doesn't support model cache!";
  uint8_t* buffer_data = buffer->data();
  // Deserialize graph_name
  size_t graph_name_size{0};
  std::memcpy(&graph_name_size, buffer_data, sizeof(graph_name_size));
  buffer_data += sizeof(graph_name_size);
  NNADAPTER_CHECK_GT(graph_name_size, 0);
  graph_name_.resize(graph_name_size);
  std::memcpy(&graph_name_[0], buffer_data, graph_name_size);
  buffer_data += graph_name_size;
  // Deserialize input_tensor_ids_
  size_t input_tensor_ids_size{0};
  std::memcpy(
      &input_tensor_ids_size, buffer_data, sizeof(input_tensor_ids_size));
  buffer_data += sizeof(input_tensor_ids_size);
  input_tensor_ids_.resize(input_tensor_ids_size / sizeof(uint32_t));
  std::memcpy(input_tensor_ids_.data(), buffer_data, input_tensor_ids_size);
  buffer_data += input_tensor_ids_size;
  // Deserialize output_tensor_ids_
  size_t output_tensor_ids_size{0};
  std::memcpy(
      &output_tensor_ids_size, buffer_data, sizeof(output_tensor_ids_size));
  buffer_data += sizeof(output_tensor_ids_size);
  output_tensor_ids_.resize(output_tensor_ids_size / sizeof(uint32_t));
  std::memcpy(output_tensor_ids_.data(), buffer_data, output_tensor_ids_size);
  buffer_data += output_tensor_ids_size;
  // Deserialize qnn_context_
  Qnn_ContextBinarySize_t binary_size =
      buffer->size() - sizeof(graph_name_size) - graph_name_size -
      sizeof(input_tensor_ids_size) - input_tensor_ids_size -
      sizeof(output_tensor_ids_size) - output_tensor_ids_size;
  QNN_CHECK(qnn_interface_.contextCreateFromBinary(
      buffer_data, binary_size, &qnn_context_, nullptr));
}

void Program::InitInputTensor(size_t index) {
  auto& tensor = input_tensors_[index];
  tensor.id = input_tensor_ids_[index];
  tensor.type = QNN_TENSOR_TYPE_APP_WRITE;
  tensor.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.dataType = ConvertToQnnDatatype(input_types_.at(index).precision);
  if (tensor.dataType == QNN_DATATYPE_UFIXED_POINT_8 ||
      tensor.dataType == QNN_DATATYPE_SFIXED_POINT_32) {
    tensor.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    tensor.quantizeParams.quantizationEncoding =
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    tensor.quantizeParams.scaleOffsetEncoding.scale =
        input_types_.at(index).asymm_per_layer_params.scale;
    tensor.quantizeParams.scaleOffsetEncoding.offset =
        -input_types_.at(index).asymm_per_layer_params.zero_point;
  }
  auto& dims = input_types_.at(index).dimensions;
  input_dims_[index].resize(dims.count);
  for (uint32_t i = 0; i < dims.count; i++) {
    input_dims_[index][i] = dims.data[i];
  }
  tensor.maxDimensions = input_dims_[index].data();
  tensor.currentDimensions = input_dims_[index].data();
  tensor.memType = QNN_TENSORMEMTYPE_RAW;
}

void Program::InitOutputTensor(size_t index) {
  auto& tensor = output_tensors_[index];
  tensor.id = output_tensor_ids_[index];
  tensor.type = QNN_TENSOR_TYPE_APP_READ;
  tensor.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.dataType = ConvertToQnnDatatype(output_types_.at(index).precision);
  if (tensor.dataType == QNN_DATATYPE_UFIXED_POINT_8 ||
      tensor.dataType == QNN_DATATYPE_SFIXED_POINT_32) {
    tensor.quantizeParams.encodingDefinition = QNN_DEFINITION_DEFINED;
    tensor.quantizeParams.quantizationEncoding =
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    tensor.quantizeParams.scaleOffsetEncoding.scale =
        output_types_.at(index).asymm_per_layer_params.scale;
    tensor.quantizeParams.scaleOffsetEncoding.offset =
        -output_types_.at(index).asymm_per_layer_params.zero_point;
  }
  auto& dims = output_types_.at(index).dimensions;
  output_dims_[index].resize(dims.count);
  for (uint32_t i = 0; i < dims.count; i++) {
    output_dims_[index][i] = dims.data[i];
  }
  tensor.maxDimensions = output_dims_[index].data();
  tensor.currentDimensions = output_dims_[index].data();
  tensor.memType = QNN_TENSORMEMTYPE_RAW;
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
