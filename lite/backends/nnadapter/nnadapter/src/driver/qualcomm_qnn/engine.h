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

#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "driver/qualcomm_qnn/converter/converter.h"
#include "driver/qualcomm_qnn/utility.h"

namespace nnadapter {
namespace qualcomm_qnn {

class Device {
 public:
  Device() {}
  ~Device() {}
};

class Context {
 public:
  explicit Context(void* device, const char* properties);
  ~Context();

  std::string RuntimeLib() { return runtime_lib_; }

 private:
  void* device_{nullptr};
  void* context_{nullptr};
  std::string runtime_lib_;
};

class Program {
 public:
  explicit Program(Context* context);
  ~Program();

  int Build(core::Model* model, core::Cache* cache);
  int Execute(uint32_t input_count,
              core::Argument* input_arguments,
              uint32_t output_count,
              core::Argument* output_arguments);

 private:
  void Clear();
  int BuildFromModel(core::Model* model);
  int BuildFromCache(core::Cache* cache);
  int CheckInputsAndOutputs(uint32_t input_count,
                            core::Argument* input_arguments,
                            uint32_t output_count,
                            core::Argument* output_arguments);

 private:
  Context* context_{nullptr};
  void* lib_backend_handle_{nullptr};
  QNN_INTERFACE_VER_TYPE qnn_interface_;
  const QnnBackend_Config_t* qnn_backend_configs_{nullptr};
  const QnnContext_Config_t* qnn_context_configs_{nullptr};
  Qnn_ContextHandle_t qnn_context_;
  const QnnGraph_Config_t* qnn_graph_config_{nullptr};
  Qnn_GraphHandle_t qnn_graph_handle_;
  // Map NNAdapter operand to qnn tensor
  std::map<core::Operand*, std::vector<Qnn_Tensor_t>> tensors_;
  std::vector<NNAdapterOperandType> input_types_;
  std::vector<NNAdapterOperandType> output_types_;
  std::vector<Qnn_Tensor_t> input_tensors_;
  std::vector<Qnn_Tensor_t> output_tensors_;
  std::vector<std::vector<uint32_t>> input_dims_;
  std::vector<std::vector<uint32_t>> output_dims_;
};

}  // namespace qualcomm_qnn
}  // namespace nnadapter
