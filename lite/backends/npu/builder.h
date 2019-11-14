// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "ai_ddk_lib/include/graph/buffer.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "ai_ddk_lib/include/graph/operator.h"
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "ai_ddk_lib/include/hiai_ir_build.h"
#include "lite/core/op_lite.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"

// Extended Ops of HIAI DDK
namespace ge {
/**
 * Pads a tensor.
 * <Input>
 *      x : the input tensor
 *      padding : the input tensor must be 2-D
 *      constant_values : constant values must be a scalar
 * <Output>
 *      output : the output tensor
 * <Attr>
 *      t_paddings : Default DT_INT32 , t_paddings must be  the same with
 * datatype of the padding
 *      mode : 0: CONSTANT, 1: REFLECT, 2: SYMMETRIC
 *      T  :  datatype of constant_values  DT_INT32:3   DT_FLOAT:0
 */
REG_OP(Pad)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32}))
    .INPUT(padding, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(constant_values, TensorType({DT_INT32, DT_FLOAT}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_INT32}))
    .ATTR(t_paddings, AttrValue::INT{3})
    .ATTR(mode, AttrValue::INT{0})
    .REQUIRED_ATTR(T, AttrValue::INT)
    .OP_END();

}  // namespace ge

namespace paddle {
namespace lite {
namespace npu {

class OpList {
 public:
  static OpList& Global() {
    static thread_local OpList x;
    return x;
  }
  void clear() { lists_.clear(); }
  void add(std::shared_ptr<ge::Operator> p) { lists_.push_back(p); }

 private:
  std::vector<std::shared_ptr<ge::Operator>> lists_;
};

// Build HIAI IR graph to om model, and store om model data into lite tensor
bool BuildModel(std::vector<ge::Operator>& inputs,   // NOLINT
                std::vector<ge::Operator>& outputs,  // NOLINT
                lite::Tensor* model_data);

std::string UniqueName(const std::string& prefix);

ge::DataType CvtPrecisionType(PrecisionType itype);

ge::Format CvtDataLayoutType(DataLayoutType itype);

ge::TensorPtr CvtTensor(Tensor* in_tensor,
                        std::vector<int64_t> out_shape = {},
                        PrecisionType in_ptype = PRECISION(kFloat),
                        DataLayoutType in_ltype = DATALAYOUT(kNCHW));

template <typename T>
ge::TensorPtr CreateTensorAndFillData(std::vector<T> data,
                                      std::vector<int64_t> shape = {},
                                      ge::Format format = ge::FORMAT_NCHW) {
  const std::type_info& info = typeid(T);
  ge::DataType type = ge::DT_FLOAT;
  if (info == typeid(float)) {
    type = ge::DT_FLOAT;
  } else if (info == typeid(int8_t)) {
    type = ge::DT_INT8;
  } else if (info == typeid(int32_t)) {
    type = ge::DT_INT32;
  } else {
    LOG(FATAL) << "[NPU] Unknow value type " << info.name();
  }
  if (shape.empty()) {
    shape = {static_cast<int64_t>(data.size())};
  } else {
    int size = 1;
    for (auto i : shape) {
      size *= i;
    }
    CHECK_EQ(data.size(), size);
  }
  ge::TensorDesc desc(ge::Shape(shape), format, type);
  ge::TensorPtr tensor = std::make_shared<ge::Tensor>();
  tensor->SetTensorDesc(desc);
  tensor->SetData(reinterpret_cast<uint8_t*>(data.data()),
                  data.size() * sizeof(T));
  return tensor;
}

template <typename T>
ge::TensorPtr CreateTensorAndFillData(T value,
                                      std::vector<int64_t> shape = {1},
                                      ge::Format format = ge::FORMAT_NCHW) {
  int64_t size = 1;
  for (auto i : shape) {
    size *= i;
  }
  std::vector<T> data(size, value);
  return CreateTensorAndFillData(data, shape, format);
}

int CvtActMode(std::string act_type);

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);

}  // namespace npu
}  // namespace lite
}  // namespace paddle
