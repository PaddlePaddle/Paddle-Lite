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

#include "lite/operators/sparse_conv_op.h"
#include <algorithm>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SparseConvOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.nonzero_weights);
  CHECK_OR_FALSE(param_.oc_nonzeros);
  CHECK_OR_FALSE(param_.diffs);
  return true;
}

inline int SparseConvOutputSize(int input_size,
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

bool SparseConvOp::InferShapeImpl() const {
  const auto in_dims = param_.x->dims();
  const auto oc = param_.oc_nonzeros->dims()[0];
  std::vector<int64_t> output_shape({in_dims[0], oc});
  auto paddings = *param_.paddings;
  auto dilations = *param_.dilations;
  for (int i = 0; i < param_.strides.size(); ++i) {
    output_shape.push_back(SparseConvOutputSize(in_dims[i + 2],
                                                1,
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

REGISTER_LITE_OP(sparse_conv2d, paddle::lite::operators::SparseConvOp);
