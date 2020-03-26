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

#include "lite/operators/expand_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ExpandOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  int expand_size = param_.expand_times.size();
  int x_dims_size = param_.X->dims().size();
  CHECK_EQ(expand_size, x_dims_size)
      << "The number of expand_times size must be qual to the rank of "
         "Input(X).";
  CHECK_LE(param_.X->dims().size(), 6)
      << "The rank of Input(X) must not be greater than 6.";
  return true;
}

bool ExpandOpLite::SmartInferShape() const {
  if (!last_input_shapes.empty()) {
    if (last_input_shapes[0] == param_.x->dims() &&
        last_input_lods[0] == param_.x->lod()) {
      param_.output->Resize(last_output_shapes[0]);
      param_.output->set_lod(last_output_lods[0]);
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

  if (!last_output_shapes.empty()) {
    last_output_shapes.clear();
    last_output_lods.clear();
  }
  last_output_shapes.push_back(param_.output->dims());
  last_output_lods.push_back(param_.output->lod());

  return true;
}

bool ExpandOpLite::InferShape() const {
  DDim out_dims(param_.X->dims());
  for (size_t i = 0; i < param_.expand_times.size(); ++i) {
    out_dims[i] *= param_.expand_times[i];
  }
  param_.Out->Resize(out_dims);
  return true;
}

bool ExpandOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto X_name = opdesc.Input("X").front();
  auto Out_name = opdesc.Output("Out").front();
  param_.X = GetVar<lite::Tensor>(scope, X_name);
  param_.Out = GetMutableVar<lite::Tensor>(scope, Out_name);
  param_.expand_times = opdesc.GetAttr<std::vector<int>>("expand_times");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(expand, paddle::lite::operators::ExpandOpLite);
