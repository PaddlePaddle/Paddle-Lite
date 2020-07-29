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
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "graph/buffer.h"
#include "graph/compatible/operator_reg.h"
#include "graph/graph.h"
#include "graph/model.h"
#include "graph/op/all_ops.h"
#include "graph/operator.h"
#include "lite/core/op_lite.h"
#include "lite/utils/macros.h"

// Extended ops based on HIAI DDK
namespace ge {
/*
 * Pads a tensor.
 * <Input>
 *    x : the input tensor
 *    padding : the input tensor must be 2-D
 *    constant_values : constant values must be a scalar
 * <Output>
 *    y : the output tensor
 * <Attr>
 *    mode : 0: CONSTANT, 1: REFLECT, 2: SYMMETRIC, 3:EDGE.
 * <Added in HiAI version>
 *    100.320.010.010
 */
REG_OP(Pad)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32}))
    .INPUT(padding, TensorType({DT_INT32}))
    .OPTIONAL_INPUT(constant_values, TensorType({DT_INT32, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32}))
    .ATTR(mode, AttrValue::INT{0})
    .OP_END()

    /*
     * The operation pads input according to the paddings and constant_values.
     * <Input>
     *    x : The input tensor.
     *    paddings : The values of paddings, as a role of dimensions to be added
     * on the input tensor x, must be a Const-OP. constant_values : A tensor of
     * the same type as x, that indicates the value to use for padding input,
     *                      must be a Const-OP.
     * <Output>
     *    y : The output tensor.
     * <Added in HiAI version>
     *    100.320.010.010
     */
    REG_OP(PadV2)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32}))
    .INPUT(paddings, TensorType({DT_INT32}))
    .INPUT(constant_values, TensorType({DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32}))
    .OP_END()

    /*
     * Computes instance norm
     * <Input>
     *    x : Input tensor which supports 4D dimension format.
     *    scale : A tesnor, multiple to result
     *    bias : A tensor, add to result
     * <Output>
     *    y : Output tensor
     * <Attr>
     *    reduction_indices : The dimensions to reduce
     *    epsilon : A very small float number used to avoid dividing by zero.
     * <Added in HiAI version>
     *    100.320.010.010
     */
    REG_OP(InstanceNorm)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(scale, TensorType({DT_FLOAT}))
    .INPUT(bias, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .REQUIRED_ATTR(reduction_indices, AttrValue::LIST_INT)
    .ATTR(epsilon, AttrValue::FLOAT{1e-7f})
    .OP_END()

    /*
     * Multiplies slices of two tensors in batches.
     * <Input>
     *    x1 : The input tensor
     *    x2 : The input tensor
     * <Output>
     *    y : The output tensor
     * <Attr>
     *    adj_x1 : adj_x1 is true, the input tensor x1  is  transposed,
     * otherwise it will not be transposed.
     *             Default is false (The current version only supports false).
     *    adj_x2 : adj_x2 is true, the input tensor x2  is  transposed,
     * otherwise it will not be transposed.
     *             Default is false.
     * <Added in HiAI version>
     *    100.320.010.010
     */
    REG_OP(BatchMatMul)
    .INPUT(x1, TensorType({DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(adj_x1, AttrValue::BOOL{false})
    .ATTR(adj_x2, AttrValue::BOOL{false})
    .OP_END()

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
                        DataLayoutType in_layout = DATALAYOUT(kNCHW));

int CvtActMode(std::string act_type);

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
