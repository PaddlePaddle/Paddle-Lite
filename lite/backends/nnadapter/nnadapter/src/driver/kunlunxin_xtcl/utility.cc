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

#include "driver/kunlunxin_xtcl/utility.h"
#include "utility/cache.h"
#include "utility/debug.h"

namespace nnadapter {
namespace kunlunxin_xtcl {

static const char* NNADAPTER_KUNLUNXIN_XTCL_CACHE_GRAPH_DATA_KEY = "graph_data";
static const char* NNADAPTER_KUNLUNXIN_XTCL_CACHE_MODULE_DATA_KEY =
    "module_data";
static const char* NNADAPTER_KUNLUNXIN_XTCL_CACHE_PARAMS_DATA_KEY =
    "params_data";

std::shared_ptr<xtcl::network::xRuntimeInstance> LoadInstanceRuntimeFromBuffer(
    int device_id, std::vector<uint8_t>* model_buffer) {
  NNADAPTER_CHECK_GE(device_id, 0);
  NNADAPTER_CHECK(model_buffer != nullptr);
  auto helper = std::make_shared<nnadapter::Cache>();
  NNADAPTER_CHECK(
      helper->Deserialize(model_buffer->data(), model_buffer->size()))
      << "Failed to deserialize a XTCL runtime instance!";
  NNADAPTER_VLOG(3) << "Deserialize a XTCL runtime instance success.";
  std::string graph_data;
  std::string module_data;
  std::string params_data;
  NNADAPTER_CHECK(
      helper->Get(NNADAPTER_KUNLUNXIN_XTCL_CACHE_GRAPH_DATA_KEY, &graph_data));
  NNADAPTER_VLOG(3) << "graph size: " << graph_data.size();
  NNADAPTER_CHECK(helper->Get(NNADAPTER_KUNLUNXIN_XTCL_CACHE_MODULE_DATA_KEY,
                              &module_data));
  NNADAPTER_VLOG(3) << "module size: " << module_data.size();
  helper->Get(NNADAPTER_KUNLUNXIN_XTCL_CACHE_PARAMS_DATA_KEY, &params_data);
  NNADAPTER_VLOG(3) << "params size: " << params_data.size();
  xtcl::xFunction func;
  xtcl::network::xTensorCompiler compiler(func);
  auto runtime_instance = compiler.CreateRuntimeInstancePtr(
      graph_data, module_data, params_data, device_id);
  NNADAPTER_CHECK(runtime_instance != nullptr)
      << "Failed to load a XTCL runtime instance from buffer!";
  NNADAPTER_VLOG(3) << "Load a XTCL runtime instance from buffer success.";
  return std::shared_ptr<xtcl::network::xRuntimeInstance>(runtime_instance);
}

std::shared_ptr<xtcl::network::xRuntimeInstance> BuildInstanceRuntimeToBuffer(
    int device_id,
    std::string device_target,
    xtcl::network::xNetworkBuilder* builder,
    xtcl::network::xTensorCompiler::ParamNDArrayMap* params,
    xtcl::Array<xtcl::xExpr>* outputs,
    std::vector<uint8_t>* model_buffer) {
  NNADAPTER_CHECK_GE(device_id, 0);
  NNADAPTER_CHECK(builder != nullptr);
  NNADAPTER_CHECK(outputs != nullptr);
  NNADAPTER_CHECK_GT(outputs->size(), 0);
  // Use TupleNode to support multiple outputs
  auto dev_tgt = xtcl::NullValue<xtcl::Target>();
  if (!device_target.empty()) {
    dev_tgt = xtcl::Target(device_target);
  }
  NNADAPTER_VLOG(3) << "Start to finalize a network.";
  xtcl::xFunction network =
      builder->FinalizeNetwork(xtcl::relay::Tuple(*outputs));
  NNADAPTER_VLOG(3)
      << "Initialize a XTCL compiler with a freezed network and device target("
      << (device_target.empty() ? "null" : device_target) << ").";
  xtcl::network::xTensorCompiler compiler(network, dev_tgt);
  compiler.SetParams(*params);  // Set the data of constant tensors
  NNADAPTER_VLOG(3) << "Start to build the network.";
  compiler.Build();
  NNADAPTER_VLOG(3) << "Start to create a XTCL runtime instance for device id("
                    << device_id << ").";
  auto runtime_instance = compiler.CreateRuntimeInstancePtr(device_id);
  NNADAPTER_CHECK(runtime_instance != nullptr)
      << "Failed to create a XTCL runtime instance from network!";
  NNADAPTER_VLOG(3) << "Create a XTCL runtime instance from network success.";
  if (model_buffer) {
    auto helper = std::make_shared<nnadapter::Cache>();
    // Dump graph, module, params to strings and serialize all of them into a
    // buffer
    auto graph_data = compiler.SerializeRuntimeGraph();
    auto module_data = compiler.SerializeRuntimeModule();
    auto params_data = compiler.SerializeRuntimeParams();
    if (!graph_data.empty()) {
      NNADAPTER_CHECK(helper->Set(NNADAPTER_KUNLUNXIN_XTCL_CACHE_GRAPH_DATA_KEY,
                                  graph_data));
    }
    NNADAPTER_VLOG(3) << "graph size: " << graph_data.size();
    if (!module_data.empty()) {
      NNADAPTER_CHECK(helper->Set(
          NNADAPTER_KUNLUNXIN_XTCL_CACHE_MODULE_DATA_KEY, module_data));
    }
    NNADAPTER_VLOG(3) << "module size: " << module_data.size();
    if (!params_data.empty()) {
      NNADAPTER_CHECK(helper->Set(
          NNADAPTER_KUNLUNXIN_XTCL_CACHE_PARAMS_DATA_KEY, params_data));
    }
    NNADAPTER_VLOG(3) << "params size: " << params_data.size();
    auto buffer_size = helper->GetSerializedSize();
    model_buffer->resize(buffer_size);
    NNADAPTER_CHECK(helper->Serialize(model_buffer->data(), buffer_size))
        << "Failed to serialize a XTCL runtime instance!";
    NNADAPTER_VLOG(3) << "Serialize a XTCL runtime instance success.";
  }
  return std::shared_ptr<xtcl::network::xRuntimeInstance>(runtime_instance);
}

xtcl::DataType ConvertToXTCLDataType(
    NNAdapterOperandPrecisionCode input_precision) {
  xtcl::DataType output_precision = ::xtcl::DataType::Float(32);
  switch (input_precision) {
    case NNADAPTER_BOOL8:
      output_precision = ::xtcl::DataType::Bool();
      break;
    case NNADAPTER_INT8:
      output_precision = ::xtcl::DataType::Int(8);
      break;
    case NNADAPTER_UINT8:
      output_precision = ::xtcl::DataType::UInt(8);
      break;
    case NNADAPTER_INT16:
      output_precision = ::xtcl::DataType::Int(16);
      break;
    case NNADAPTER_UINT16:
      output_precision = ::xtcl::DataType::UInt(16);
      break;
    case NNADAPTER_INT32:
      output_precision = ::xtcl::DataType::Int(32);
      break;
    case NNADAPTER_UINT32:
      output_precision = ::xtcl::DataType::UInt(32);
      break;
    case NNADAPTER_INT64:
      output_precision = ::xtcl::DataType::Int(64);
      break;
    case NNADAPTER_UINT64:
      output_precision = ::xtcl::DataType::UInt(64);
      break;
    case NNADAPTER_FLOAT16:
      output_precision = ::xtcl::DataType::Float(16);
      break;
    case NNADAPTER_FLOAT32:
      output_precision = ::xtcl::DataType::Float(32);
      break;
    case NNADAPTER_FLOAT64:
      output_precision = ::xtcl::DataType::Float(64);
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to xtcl:DataType !";
      break;
  }
  return output_precision;
}

DLDataType ConvertToDLDataType(NNAdapterOperandPrecisionCode input_precision) {
  DLDataType output_precision = {kDLFloat, 32, 1};
  switch (input_precision) {
    case NNADAPTER_INT8:
      output_precision = {kDLInt, 8, 1};
      break;
    case NNADAPTER_INT16:
      output_precision = {kDLInt, 16, 1};
      break;
    case NNADAPTER_INT32:
      output_precision = {kDLInt, 32, 1};
      break;
    case NNADAPTER_INT64:
      output_precision = {kDLInt, 64, 1};
      break;
    case NNADAPTER_FLOAT32:
      output_precision = {kDLFloat, 32, 1};
      break;
    default:
      NNADAPTER_LOG(FATAL)
          << "Failed to convert the NNAdapter operand precision code("
          << OperandPrecisionCodeToString(input_precision)
          << ") to DLDataType !";
      break;
  }
  return output_precision;
}

xtcl::xNDArray CreateXTCLNDArray(std::vector<int64_t> shape,
                                 DLDataType dtype,
                                 const void* buffer) {
  NNADAPTER_CHECK(buffer != nullptr);
  int64_t size = dtype.bits / 8;
  for (auto dim : shape) {
    size *= dim;
  }
  auto ndarray = xtcl::xNDArray::Empty(shape, dtype, {kDLCPU, 0});
  ndarray.CopyFromBytes(buffer, size);
  return ndarray;
}

}  // namespace kunlunxin_xtcl
}  // namespace nnadapter
