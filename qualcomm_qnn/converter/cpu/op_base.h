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

#include <string>
#include <vector>

#include "CPU/QnnCpuOpPackage.h"
#include "driver/qualcomm_qnn/utility.h"

namespace nnadapter {
namespace qualcomm_qnn {
namespace cpu {

class OpBase {
 public:
  OpBase() { is_finalize_ = false; }
  OpBase(const char* name, const char* typeName)
      : name_(name), type_name_(typeName) {
    is_finalize_ = false;
  }
  virtual ~OpBase() = default;

  virtual Qnn_ErrorHandle_t Finalize() { return QNN_OP_PACKAGE_ERROR_GENERAL; }

  virtual Qnn_ErrorHandle_t Execute() { return QNN_OP_PACKAGE_ERROR_GENERAL; }

  virtual Qnn_ErrorHandle_t SetOpNode(QnnCpuOpPackage_Node_t* node) {
    return QNN_OP_PACKAGE_ERROR_GENERAL;
  }

  Qnn_ErrorHandle_t AddInput(QnnCpuOpPackage_Tensor_t* in_tensor) {
    input_tensor_.emplace_back(in_tensor);
    return QNN_SUCCESS;
  }

  Qnn_ErrorHandle_t AddOutput(QnnCpuOpPackage_Tensor_t* out_tensor) {
    output_tensor_.emplace_back(out_tensor);
    return QNN_SUCCESS;
  }

  std::string GetName() { return name_; }

  std::string GetTypeName() { return type_name_; }

  QnnCpuOpPackage_Tensor_t* GetInput(uint32_t index) {
    return input_tensor_[index];
  }

  QnnCpuOpPackage_Tensor_t* GetOutput(uint32_t index) {
    return output_tensor_[index];
  }

  uint32_t InputSize() { return input_tensor_.size(); }

  uint32_t OutputSize() { return output_tensor_.size(); }

  uint32_t TensorRank(QnnCpuOpPackage_Tensor_t* tensor) { return tensor->rank; }

  uint32_t TensorSize(QnnCpuOpPackage_Tensor_t* tensor) {
    uint32_t size = 1;
    for (uint32_t i = 0; i < TensorRank(tensor); i++) {
      size *= tensor->currentDimensions[i];
    }
    return size;
  }

  void SetIsFinalize(bool is_finalize) { is_finalize_ = is_finalize; }

  bool GetIsFinalize() { return is_finalize_; }

 private:
  std::string name_;
  std::string type_name_;
  bool is_finalize_;
  std::vector<QnnCpuOpPackage_Tensor_t*> input_tensor_;
  std::vector<QnnCpuOpPackage_Tensor_t*> output_tensor_;
};

}  // namespace cpu
}  // namespace qualcomm_qnn
}  // namespace nnadapter
