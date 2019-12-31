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

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "graph/buffer.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "graph/op/all_ops.h"
#include "graph/operator.h"
#include "graph/operator_reg.h"
#include "lite/core/op_lite.h"
#include "lite/utils/macros.h"

// Extended ops based on HIAI DDK
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
namespace subgraph {
namespace npu {

// Type/tensor converters for converting Paddle type/tensor to HiAI type/tensor
bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);

ge::DataType CvtPrecisionType(PrecisionType itype);

ge::Format CvtDataLayoutType(DataLayoutType itype);

// Padding the shape to 4-dimensions(NCHW) for HiAI
std::vector<int64_t> CvtShape(const std::vector<int64_t>& in_shape);

std::vector<int64_t> CvtShape(const DDim& in_dims);

ge::TensorPtr CvtTensor(const Tensor& in_tensor,
                        std::vector<int64_t> out_shape = {},
                        PrecisionType in_precision = PRECISION(kFloat),
                        DataLayoutType in_layout = DATALAYOUT(kNCHW));

int CvtActMode(std::string act_type);

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
