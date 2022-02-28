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

#include "driver/nvidia_tensorrt/converter/converter.h"
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {
namespace nvidia_tensorrt {

#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  extern int __func_name__(Converter* converter, core::Operation* operation);
#include "driver/nvidia_tensorrt/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_NVIDIA_TENSORRT_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

nvinfer1::INetworkDefinition* Converter::network() { return network_; }

int Converter::Apply(core::Model* model) {
  // Convert the NNAdapter operations to trt nodes
  std::vector<core::Operation*> operations =
      SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    NNADAPTER_VLOG(5) << "Converting " << OperationTypeToString(operation->type)
                      << " ...";
    switch (operation->type) {
#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  case NNADAPTER_##__op_type__:                        \
    __func_name__(this, operation);                    \
    break;
#include "driver/nvidia_tensorrt/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_NVIDIA_TENSORRT_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER
      default:
        NNADAPTER_LOG(FATAL) << "Unsupported operation("
                             << OperationTypeToString(operation->type)
                             << ") is found.";
        break;
    }
  }
  return NNADAPTER_NO_ERROR;
}

nvinfer1::ITensor* Converter::GetMappedTensor(core::Operand* operand) {
  auto it = tensors_->find(operand);
  if (it != tensors_->end()) {
    return it->second.back();
  }
  return nullptr;
}

void Converter::UpdateTensorMap(core::Operand* operand,
                                nvinfer1::ITensor* tensor) {
  if (tensors_->count(operand) > 0) {
    tensors_->at(operand).push_back(tensor);
  } else {
    tensors_->emplace(operand, std::vector<nvinfer1::ITensor*>{tensor});
  }
}

nvinfer1::ITensor* Converter::ConvertOperand(
    core::Operand* operand, const std::vector<int32_t>& dimensions) {
  NNADAPTER_CHECK(IsModelInputOperand(operand));
  nvinfer1::Dims dims;
  if (dimensions.empty()) {
    dims.nbDims = operand->type.dimensions.count;
    memcpy(
        dims.d, operand->type.dimensions.data, dims.nbDims * sizeof(int32_t));
  } else {
    dims.nbDims = dimensions.size();
    memcpy(dims.d, dimensions.data(), dims.nbDims * sizeof(int32_t));
  }
  std::string name = "data" + std::to_string(input_index_);
  input_index_++;
  auto precision = ConvertToNVDataType(operand->type.precision);
  nvinfer1::ITensor* data = network_->addInput(name.c_str(), precision, dims);
  NNADAPTER_CHECK(data);
  UpdateTensorMap(operand, data);
  return data;
}

nvinfer1::Weights Converter::OperandToWeights(core::Operand* operand) {
  NNADAPTER_CHECK(operand);
  NNADAPTER_CHECK(IsConstantOperand(operand));
  nvinfer1::Weights weight;
  weight.type = ConvertToNVDataType(operand->type.precision);
  weight.values = operand->buffer;
  weight.count =
      operand->length / GetOperandPrecisionDataLength(operand->type.precision);
  return weight;
}

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
