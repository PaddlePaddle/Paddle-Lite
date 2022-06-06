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
#include "driver/qualcomm_qnn/utility.h"

namespace nnadapter {
namespace qualcomm_qnn {

class Converter {
 public:
  explicit Converter(
      QNN_INTERFACE_VER_TYPE qnn_interface,
      Qnn_GraphHandle_t* qnn_graph,
      std::map<core::Operand*, std::vector<Qnn_Tensor_t>>* tensors)
      : qnn_interface_(qnn_interface),
        qnn_graph_(qnn_graph),
        tensors_(tensors),
        tensor_indexes_(0) {}
  ~Converter() {}

  // Convert a NNAdapter model to a trt network
  int Apply(core::Model* model);

  Qnn_Tensor_t ConvertOperand(core::Operand* operand,
                              const std::vector<int32_t>& dimensions = {});

  Qnn_Tensor_t GetMappedTensor(core::Operand* operand);

  template <typename T>
  Qnn_Param_t GetParam(const char* name, const T data);

  template <typename T>
  Qnn_Param_t GetParam(const char* name,
                       std::vector<T> data,
                       std::vector<uint32_t> dims = {});

  void AddNode(const char* op_type,
               std::vector<Qnn_Tensor_t> input_tensors,
               std::vector<Qnn_Tensor_t> output_tensors,
               std::vector<Qnn_Param_t> params = {});

 private:
  QNN_INTERFACE_VER_TYPE qnn_interface_;
  Qnn_GraphHandle_t* qnn_graph_{nullptr};
  std::map<core::Operand*, std::vector<Qnn_Tensor_t>>* tensors_{nullptr};
  uint32_t tensor_indexes_{0};
  uint32_t op_indexes_{0};
  std::vector<std::vector<uint32_t>> dims_;
};

}  // namespace qualcomm_qnn
}  // namespace nnadapter
