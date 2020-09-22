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

#include "lite/operators/conv_op.h"
#include <algorithm>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ConvOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.filter);
  // bias is optional.

  const auto in_dims = param_.x->dims();
  const auto filter_dims = param_.filter->dims();

  CHECK_OR_FALSE(in_dims.size() == 4 || in_dims.size() == 5);

  CHECK_EQ_OR_FALSE(in_dims.size(), filter_dims.size());
  CHECK_OR_FALSE(in_dims.size() - param_.strides.size() == 2U);
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

void UpdatePaddingAndDilation(std::vector<int>* paddings,
                              std::vector<int>* dilations,
                              const std::vector<int>& strides,
                              const std::string padding_algorithm,
                              const lite::DDim data_dims,
                              const lite::DDim& ksize) {
  // when padding_desc is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (size_t i = 0; i < strides.size(); ++i) {
      int out_size = (data_dims[i + 2] + strides[i] - 1) / strides[i];
      int pad_sum = (std::max)(
          (out_size - 1) * strides[i] + ksize[i + 2] - data_dims[i + 2],
          (int64_t)0);
      int pad_0 = pad_sum / 2;
      int pad_1 = pad_sum - pad_0;
      // pad
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
      // dilation
      *(dilations->begin() + i) = 1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto& it : *paddings) {
      it = 0;
    }
  }
}

bool ConvOpLite::InferShapeImpl() const {
  const auto in_dims = param_.x->dims();
  const auto filter_dims = param_.filter->dims();

  UpdatePaddingAndDilation(param_.paddings.get(),
                           param_.dilations.get(),
                           param_.strides,
                           padding_algorithm_,
                           in_dims,
                           filter_dims);
  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  auto paddings = *param_.paddings;
  auto dilations = *param_.dilations;
  for (size_t i = 0; i < param_.strides.size(); ++i) {
    output_shape.push_back(ConvOutputSize(in_dims[i + 2],
                                          filter_dims[i + 2],
                                          dilations[i],
                                          paddings[i * 2],
                                          paddings[i * 2 + 1],
                                          param_.strides[i]));
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

REGISTER_LITE_OP(conv2d, paddle::lite::operators::ConvOpLite);
REGISTER_LITE_OP(depthwise_conv2d, paddle::lite::operators::ConvOpLite);
