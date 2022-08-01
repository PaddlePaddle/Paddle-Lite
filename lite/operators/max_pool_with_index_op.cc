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

#include "lite/operators/max_pool_with_index_op.h"
#include <algorithm>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MaxPoolWithIndexOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);

  const auto& x_dims = param_.x->dims();
  const auto& strides = param_.strides;
  const auto& ksize = param_.ksize;
  const auto& paddings = *param_.paddings;
  // "Pooling intput should be 4-D or 5-D tensor."
  CHECK_OR_FALSE(x_dims.size() == 4 || x_dims.size() == 5);
  // Input size and pooling size should be consistent.
  CHECK_OR_FALSE(x_dims.size() - ksize.size() == 2U);
  // Strides size and pooling size should be the same.
  CHECK_OR_FALSE(ksize.size() == strides.size());
  // Paddings size must be 4.
  CHECK_OR_FALSE(paddings.size() == 4L);

  return true;
}

inline int MaxPoolOutputSize(int input_size,
                             int filter_size,
                             int padding,
                             int stride) {
  int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  return output_size;
}

bool MaxPoolWithIndexOpLite::InferShapeImpl() const {
  const auto x_dims = param_.x->dims();
  const auto ksize = param_.ksize;
  std::vector<int64_t> output_shape({x_dims[0], x_dims[1]});
  const auto& strides = param_.strides;
  const auto& paddings = *param_.paddings;
  const auto adaptive = param_.adaptive;

  if (adaptive) {
    output_shape.insert(output_shape.end(), ksize.begin(), ksize.end());
  } else {
    for (size_t i = 0; i < ksize.size(); ++i) {
      output_shape.push_back(
          MaxPoolOutputSize(x_dims[i + 2], ksize[i], paddings[i], strides[i]));
    }
  }
  param_.output->Resize(lite::DDim(output_shape));
  param_.mask->Resize(lite::DDim(output_shape));

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(max_pool2d_with_index,
                 paddle::lite::operators::MaxPoolWithIndexOpLite);
