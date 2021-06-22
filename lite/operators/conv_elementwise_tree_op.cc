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

#include "lite/operators/conv_elementwise_tree_op.h"
#include <algorithm>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ConvElementwiseTreeOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.y);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.filter);
  // bias is optional.

  const auto x_dims = param_.x->dims();
  const auto y_dims = param_.y->dims();
  const auto filter_dims = param_.filter->dims();

  CHECK_EQ_OR_FALSE(x_dims.size(), y_dims.size());
  CHECK_EQ_OR_FALSE(x_dims.size(), filter_dims.size());
  CHECK_OR_FALSE(x_dims.size() - param_.conv_param.strides.size() == 2U);
  CHECK_EQ_OR_FALSE(filter_dims.size(), 4UL);

  return true;
}

inline int ConvOutputSize(int input_size,
                          int filter_size,
                          int dilation,
                          int pad_left,
                          int pad_right,
                          int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size =
      (input_size + (pad_left + pad_right) - dkernel) / stride + 1;

  return output_size;
}

bool ConvElementwiseTreeOpLite::InferShapeImpl() const {
  const auto in_dims = param_.x->dims();
  const auto filter_dims = param_.filter->dims();
  const auto y_dims = param_.y->dims();

  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  auto paddings = *param_.conv_param.paddings;
  auto dilations = *param_.conv_param.dilations;
  for (size_t i = 0; i < param_.conv_param.strides.size(); ++i) {
    output_shape.push_back(ConvOutputSize(in_dims[i + 2],
                                          filter_dims[i + 2],
                                          dilations[i],
                                          paddings[i * 2],
                                          paddings[i * 2 + 1],
                                          param_.conv_param.strides[i]));
  }

  // check output_shape equal input_shape
  for (int i = 0; i < 4; i++) {
    if (output_shape[i] != y_dims[i]) {
      LOG(INFO) << "output_shape: " << output_shape[i] << ", i: " << i
                << " doesn't equal to y_dims: " << y_dims;
      return false;
    }
  }

  // Set output dims
  param_.output->Resize(lite::DDim(output_shape));
  // share LoD
  param_.output->set_lod(param_.x->lod());

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(conv_elementwise_tree,
                 paddle::lite::operators::ConvElementwiseTreeOpLite);
