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

#include "driver/qualcomm_qnn/converter/converter.h"

namespace nnadapter {
namespace qualcomm_qnn {

#define REGISTER_CONVERTER(__op_type__, __func_name__) \
  extern int __func_name__(Converter* converter, core::Operation* operation);
#include "driver/qualcomm_qnn/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_QUALCOMM_QNN_CONVERTER_ALL_H__
#undef REGISTER_CONVERTER

int Converter::Apply(core::Model* model) {
  // Convert the NNAdapter operations to qnn nodes
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
#include "driver/qualcomm_qnn/converter/all.h"  // NOLINT
#undef __NNADAPTER_DRIVER_QUALCOMM_QNN_CONVERTER_ALL_H__
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

Qnn_Tensor_t Converter::ConvertOperand(core::Operand* operand,
                                       const std::vector<int32_t>& dimensions) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.id = tensor_indexes_++;
  if (IsModelInputOperand(operand)) {
    tensor.type = QNN_TENSOR_TYPE_APP_WRITE;
  } else if (IsModelOutputOperand(operand)) {
    tensor.type = QNN_TENSOR_TYPE_APP_READ;
  } else if (IsConstantOperand(operand)) {
    tensor.type = QNN_TENSOR_TYPE_STATIC;
  } else {
    tensor.type = QNN_TENSOR_TYPE_NATIVE;
  }
  tensor.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.dataType = ConvertToQnnDatatype(operand->type.precision);
  if (tensor.dataType == QNN_DATATYPE_UFIXED_POINT_8) {
    tensor.quantizeParams.encodingDefinition = QNN_DEFINITION_IMPL_GENERATED;
    tensor.quantizeParams.quantizationEncoding =
        QNN_QUANTIZATION_ENCODING_SCALE_OFFSET;
    tensor.quantizeParams.scaleOffsetEncoding.scale =
        operand->type.asymm_per_layer_params.scale;
    tensor.quantizeParams.scaleOffsetEncoding.offset =
        operand->type.asymm_per_layer_params.zero_point;
  }
  auto& dims = operand->type.dimensions;
  tensor.rank = dims.count;
  std::vector<uint32_t> dims_data;
  for (uint32_t i = 0; i < dims.count; i++) {
    dims_data.push_back(dims.data[i]);
  }
  tensor.maxDimensions = dims_data.data();
  tensor.currentDimensions = dims_data.data();
  if (IsConstantOperand(operand)) {
    tensor.clientBuf.dataSize = operand->length;
    tensor.clientBuf.data = operand->buffer;
  }
  tensor.memType = QNN_TENSORMEMTYPE_RAW;
  QNN_CHECK(qnn_interface_.tensorCreateGraphTensor(*qnn_graph_, tensor));
  return tensor;
}

Qnn_Tensor_t Converter::GetMappedTensor(core::Operand* operand) {
  if (tensors_->count(operand)) {
    return tensors_->at(operand).back();
  } else {
    auto tensor = ConvertOperand(operand);
    tensors_->emplace(operand, std::vector<Qnn_Tensor_t>{tensor});
    return tensor;
  }
}

template <>
Qnn_Param_t Converter::GetParam<uint32_t>(const char* name,
                                          const uint32_t data) {
  Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
  scalar.dataType = QNN_DATATYPE_UINT_32;
  scalar.uint32Value = data;
  Qnn_Param_t param = QNN_PARAM_INIT;
  param.name = name;
  param.paramType = QNN_PARAMTYPE_SCALAR;
  param.scalarParam = scalar;
  return param;
}

template <>
Qnn_Param_t Converter::GetParam<float>(const char* name, const float data) {
  Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
  scalar.dataType = QNN_DATATYPE_FLOAT_32;
  scalar.floatValue = data;
  Qnn_Param_t param = QNN_PARAM_INIT;
  param.name = name;
  param.paramType = QNN_PARAMTYPE_SCALAR;
  param.scalarParam = scalar;
  return param;
}

template <>
Qnn_Param_t Converter::GetParam<bool>(const char* name, const bool data) {
  Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
  scalar.dataType = QNN_DATATYPE_BOOL_8;
  scalar.bool8Value = data;
  Qnn_Param_t param = QNN_PARAM_INIT;
  param.name = name;
  param.paramType = QNN_PARAMTYPE_SCALAR;
  param.scalarParam = scalar;
  return param;
}

template <typename T>
Qnn_Param_t Converter::GetParam(const char* name,
                                std::vector<T> data,
                                std::vector<uint32_t> dims) {
  Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
  tensor.id = tensor_indexes_++;
  tensor.type = QNN_TENSOR_TYPE_STATIC;
  tensor.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
  tensor.dataType = GetQnnDatatype<T>();
  if (dims.empty()) {
    dims.push_back(data.size());
  }
  tensor.rank = dims.size();
  tensor.maxDimensions = dims.data();
  tensor.currentDimensions = dims.data();
  tensor.clientBuf.dataSize = data.size() * sizeof(T);
  tensor.clientBuf.data = data.data();
  tensor.memType = QNN_TENSORMEMTYPE_RAW;
  QNN_CHECK(qnn_interface_.tensorCreateGraphTensor(*qnn_graph_, tensor));

  Qnn_Param_t param = QNN_PARAM_INIT;
  param.name = name;
  param.paramType = QNN_PARAMTYPE_TENSOR;
  param.tensorParam = tensor;
  return param;
}

template Qnn_Param_t Converter::GetParam<float>(const char* name,
                                                std::vector<float> data,
                                                std::vector<uint32_t> dims);

template Qnn_Param_t Converter::GetParam<uint32_t>(const char* name,
                                                   std::vector<uint32_t> data,
                                                   std::vector<uint32_t> dims);

void Converter::AddNode(const char* op_type,
                        std::vector<Qnn_Tensor_t> input_tensors,
                        std::vector<Qnn_Tensor_t> output_tensors,
                        std::vector<Qnn_Param_t> params) {
  Qnn_OpConfig_t op_def = QNN_OPCONFIG_INIT;
  std::string name(op_type);
  name += "_" + std::to_string(reinterpret_cast<uint64_t>(op_type));
  op_def.name = name.c_str();
  op_def.packageName = QNN_OP_PACKAGE_NAME_QTI_AISW;
  op_def.typeName = op_type;
  op_def.numOfInputs = input_tensors.size();
  op_def.inputTensors = input_tensors.data();
  op_def.numOfOutputs = output_tensors.size();
  op_def.outputTensors = output_tensors.data();
  op_def.numOfParams = params.size();
  op_def.params = params.data();
  QNN_CHECK(qnn_interface_.graphAddNode(*qnn_graph_, op_def));
}

}  // namespace qualcomm_qnn
}  // namespace nnadapter
