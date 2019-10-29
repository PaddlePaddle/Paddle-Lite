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
 * Multiply the matrix x1 by the matrix x2 to generate x1 * x2.
 * The inputs must be two-dimensional matrices and the inner dimension of "x1"
 * (after being transposed if transpose_x1 is true) must match the outer
 * dimension of "x2" (after being transposed if transposed_x2 is true). <Input>
 *      x : the first input tensor, must be non const op.
 *      w : the second input tensor, must be const op.
 *      bias: the optional bias tensor, must be const op.
 * <Output>
 *      y : the output tensor.
 * <Attr>
 *      has_bias: If true, enable input bias.
 */
REG_OP(MatMul)
    .INPUT(x, TensorType({DT_FLOAT}))
    .INPUT(w, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))  // bias must be const input
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(has_bias, AttrValue::BOOL{false})  // when has input::bias,set true
    .OP_END()

    /**
     * Computes the gradients of convolution with respect to the input.
     * <Input>
     *      input_sizes : An integer vector representing the shape of input,
     * where input is a 4-D [batch, height, width, channels] tensor.
     *      filter : the filter tensor, with shape [H , W, filter_channel,
     * filter_number], filter_channel must be same as x channel.
     *      x :  The input tensor.
     * <Output>
     *      y : The output tensor.
     * <Attr>
     *      format: 0: NCHW. 1: NHWC
     *      group : 1: default
     *      num_output : 0: default, num_output must be equal to
     * (filter_channel * group)
     *      pad : Padding for the beginning and ending along each axis
     *      stride : Stride along each axis.
     *      dilation : dilation value along each axis of the filter.
     *      pad_mode : 0:NOTSET, 5:VALID 6:SAME. defaul value is 0:NOTSET
     *      bias_term : 0: default
     *      kernel : The shape of the convolution kernel
     */
    REG_OP(Deconvolution)
    .INPUT(input_sizes, TensorType({DT_UINT8}))
    .INPUT(filter, TensorType({DT_FLOAT}))
    .INPUT(x, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT}))
    .ATTR(mode, AttrValue::INT{1})
    .ATTR(format, AttrValue::INT{1})
    .ATTR(group, AttrValue::INT{1})
    .ATTR(num_output, AttrValue::INT{0})
    .ATTR(pad, AttrValue::LIST_INT({0, 0, 0, 0}))
    .ATTR(stride, AttrValue::LIST_INT({1, 1}))
    .ATTR(dilation, AttrValue::LIST_INT({1, 1}))
    .ATTR(pad_mode, AttrValue::INT{0})
    .ATTR(bias_term, AttrValue::INT{0})
    .ATTR(kernel, AttrValue::LIST_INT({0, 0}))
    .OP_END()

    /**
     * Resize images to size using bilinear interpolation.
     * <Input>
     *      x : The tensor of 4-D
     *      w : A int32 Tensor of 2 elements: [height, width].
     * <Output>
     *      y : the output tensor
     * <Attr>
     *      align_corners : If true, the centers of the 4 corner pixels of the
     * input and output tensors are aligned, preserving the values at the corner
     * pixels.
     *      output_dim_mode : Defaults 2, including 0: zoom_factor , 1:
     * shrink_factor, 2: height/width. when output_dim_mode=2, the output-dim is
     * controled by the [height, width] of w.
     *      shrink_factor : shrink factor.
     *      zoom_factor : zoom factor.
     *      pad_begin : begin of pad.
     *      pad_end : end of pad.
     */
    REG_OP(ResizeBilinear)
    .INPUT(x, TensorType({DT_FLOAT, DT_INT32}))
    .INPUT(w, TensorType({DT_FLOAT, DT_INT32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_INT32}))
    .ATTR(align_corners, AttrValue::BOOL{false})
    .ATTR(output_dim_mode, AttrValue::INT{2})
    .ATTR(shrink_factor, AttrValue::INT{1})
    .ATTR(zoom_factor, AttrValue::INT{1})
    .ATTR(pad_begin, AttrValue::INT{0})
    .ATTR(pad_end, AttrValue::INT{0})
    .OP_END()

    /**
     * Resize images to size using nearest neighbor interpolation.
     * <Input>
     *      image : Resize images to size using nearest neighbor interpolation.
     *      size : Must be one dimension and two  elements
     * <Output>
     *      output : the output tensor
     * <Attr>
     *      align_corners : If true, the centers of the 4 corner pixels of the
     * input and output tensors are aligned, preserving the values at the corner
     * pixels. Defaults to false
     */
    REG_OP(ResizeNearestNeighbor)
    .INPUT(image, TensorType({DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL}))
    .INPUT(size, TensorType({DT_INT32}))
    .OUTPUT(output, TensorType({DT_FLOAT, DT_INT32, DT_UINT8, DT_BOOL}))
    .ATTR(align_corners, AttrValue::BOOL{false})
    .OP_END()

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
    .OP_END()

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

ge::DataType PrecisionConverter(PrecisionType itype);

ge::Format DataLayoutConverter(DataLayoutType itype);

ge::TensorPtr CvtFromLiteTensor(Tensor* in_tensor,
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
    LOG(FATAL) << "Unknow value type " << info.name();
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

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname);

}  // namespace npu
}  // namespace lite
}  // namespace paddle
