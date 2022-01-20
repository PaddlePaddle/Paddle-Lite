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

#include "lite/operators/unfold_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool UnfoldOpLite::CheckShape() const {
  CHECK(param_.X);
  CHECK(param_.Y);

  const auto x_dims = param_.X->dims();
  CHECK_EQ(x_dims.size(), 4);
  return true;
}

inline int CalcOutputSize(int input_size,
                          int filter_size,
                          int dilation,
                          int padding1,
                          int padding2,
                          int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + padding1 + padding2 - dkernel) / stride + 1;
  return output_size;
}

bool UnfoldOpLite::InferShapeImpl() const {
  const auto x_dims = param_.X->dims();
  std::vector<int> kernel_sizes = param_.kernel_sizes;
  std::vector<int> strides = param_.strides;
  std::vector<int> paddings = param_.paddings;
  std::vector<int> dilations = param_.dilations;
  CHECK_EQ(kernel_sizes.size(), 2);
  CHECK_EQ(strides.size(), 2);
  CHECK_EQ(paddings.size(), 4);
  CHECK_EQ(dilations.size(), 2);
  int output_channels = x_dims[1] * kernel_sizes[0] * kernel_sizes[1];
  std::vector<int64_t> output_shape({x_dims[0], output_channels});

  int output_height = CalcOutputSize(x_dims[2],
                                     kernel_sizes[0],
                                     dilations[0],
                                     paddings[0],
                                     paddings[2],
                                     strides[0]);
  int output_width = CalcOutputSize(x_dims[3],
                                    kernel_sizes[1],
                                    dilations[1],
                                    paddings[1],
                                    paddings[3],
                                    strides[1]);
  CHECK_GT(output_height, 0);
  CHECK_GT(output_width, 0);
  int output_col_length = output_height * output_width;
  output_shape.push_back(output_col_length);

  param_.Y->Resize(lite::DDim(output_shape));
  return true;
}

bool UnfoldOpLite::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  auto X_name = op_desc.Input("X").front();
  auto Y_name = op_desc.Output("Y").front();
  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Y = GetMutableVar<lite::Tensor>(scope, Y_name);
  CHECK(param_.X);
  CHECK(param_.Y);
  param_.kernel_sizes = op_desc.GetAttr<std::vector<int>>("kernel_sizes");
  param_.strides = op_desc.GetAttr<std::vector<int>>("strides");
  param_.paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  param_.dilations = op_desc.GetAttr<std::vector<int>>("dilations");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(unfold, paddle::lite::operators::UnfoldOpLite);
