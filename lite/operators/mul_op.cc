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

#include "lite/operators/mul_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MulOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.y);
  CHECK_OR_FALSE(param_.output);

  // bias is optional.

  const auto x_dims = param_.x->dims();
  const auto y_dims = param_.y->dims();

  CHECK_GT_OR_FALSE(x_dims.size(), static_cast<size_t>(param_.x_num_col_dims));
  CHECK_GT_OR_FALSE(y_dims.size(), static_cast<size_t>(param_.y_num_col_dims));

  return true;
}

bool MulOpLite::SmartInferShape() {
  if (!last_input_shapes.empty()) {
    if (last_input_shapes[0] == param_.x->dims() &&
        last_input_shapes[1] == param_.y->dims() &&
        last_input_lods[0] == param_.x->lod() &&
        last_input_lods[1] == param_.y->lod()) {
      param_.Out->Resize(last_output_shapes[0]);
      param_.Out->set_lod(last_output_lods[0]);
      return true;
    }
  }

  this->InferShape();

  if (!last_input_shapes.empty()) {
    last_input_shapes.clear();
    last_input_lods.clear();
  }

  last_input_shapes.push_back(param_.x->dims());
  last_input_lods.push_back(param_.x->lod());
  last_input_shapes.push_back(param_.y->dims());
  last_input_lods.push_back(param_.y->lod());

  if (!last_output_shapes.empty()) {
    last_output_shapes.clear();
    last_output_lods.clear();
  }
  last_output_shapes.push_back(param_.Out->dims());
  last_output_lods.push_back(param_.Out->lod());
  return true;
}

bool MulOpLite::InferShape() const {
  const auto x_dims = param_.x->dims();
  const auto y_dims = param_.y->dims();

  // Set output dims
  std::vector<int64_t> out_dims;
  for (int i = 0; i < param_.x_num_col_dims; ++i) {
    out_dims.push_back(x_dims[i]);
  }

  for (auto i = static_cast<size_t>(param_.y_num_col_dims); i < y_dims.size();
       ++i) {
    out_dims.push_back(y_dims[i]);
  }
  param_.output->Resize(lite::DDim(out_dims));
  auto out_lod = param_.output->mutable_lod();
  *out_lod = param_.x->lod();

  // share LoD
  // param_.output->set_lod(param_.input->lod());
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(mul, paddle::lite::operators::MulOpLite);
