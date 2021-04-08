// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/correlation_op.h"
#include <cmath>

namespace paddle {
namespace lite {
namespace operators {

std::vector<int64_t> CorrelationOutputSize(int batch,
                                           int input_height,
                                           int input_width,
                                           int stride1,
                                           int stride2,
                                           int kernel_size,
                                           int pad_size,
                                           int max_displacement) {
  std::vector<int64_t> output_shape({batch});
  int kernel_radius = (kernel_size - 1) / 2;
  int border_radius = kernel_radius + max_displacement;
  int padded_input_height = input_height + 2 * pad_size;
  int padded_input_width = input_width + 2 * pad_size;
  int output_channel = ((max_displacement / stride2) * 2 + 1) *
                       ((max_displacement / stride2) * 2 + 1);
  output_shape.push_back(output_channel);
  int output_height =
      std::ceil(static_cast<float>(padded_input_height - 2 * border_radius) /
                static_cast<float>(stride1));
  int output_width =
      std::ceil(static_cast<float>(padded_input_width - 2 * border_radius) /
                static_cast<float>(stride1));
  output_shape.push_back(output_height);
  output_shape.push_back(output_width);
  return output_shape;
}

bool CorrelationOp::CheckShape() const {
  CHECK(param_.input1);
  CHECK(param_.input2);
  CHECK(param_.output);

  auto x_dims = param_.input1->dims();
  CHECK_EQ(x_dims.size(), 4UL)
      << "Input(X) of CorrelationOp must be 4 dims. But received dims is: "
      << x_dims;
  auto y_dims = param_.input2->dims();
  CHECK_EQ(y_dims.size(), 4UL)
      << "Input(Y) of CorrelationOp must be 4 dims. But received dims is: "
      << x_dims;
  return true;
}

bool CorrelationOp::InferShapeImpl() const {
  auto in_dims = param_.input1->dims();
  auto out_shape = CorrelationOutputSize(in_dims[0],
                                         in_dims[2],
                                         in_dims[3],
                                         param_.stride1,
                                         param_.stride2,
                                         param_.kernel_size,
                                         param_.pad_size,
                                         param_.max_displacement);
  param_.output->Resize(out_shape);
  return true;
}

bool CorrelationOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  param_.input1 = scope->FindTensor(op_desc.Input("Input1").front());
  param_.input2 = scope->FindTensor(op_desc.Input("Input2").front());
  param_.output = scope->FindMutableTensor(op_desc.Output("Output").front());
  param_.pad_size = op_desc.GetAttr<int>("pad_size");
  param_.kernel_size = op_desc.GetAttr<int>("kernel_size");
  param_.max_displacement = op_desc.GetAttr<int>("max_displacement");
  param_.stride1 = op_desc.GetAttr<int>("stride1");
  param_.stride2 = op_desc.GetAttr<int>("stride2");
  param_.corr_type_multiply = op_desc.GetAttr<int>("corr_type_multiply");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(correlation, paddle::lite::operators::CorrelationOp);
