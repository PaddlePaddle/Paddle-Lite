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

#include "lite/operators/pad2d_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool Pad2dOpLite::CheckShape() const {
  CHECK_GT_OR_FALSE(param_.X->dims().size(), 1UL);
  CHECK_OR_FALSE(param_.Out);
  CHECK(param_.mode == "constant" || param_.mode == "reflect" ||
        param_.mode == "edge")
      << "Invilid mode.";
  CHECK_EQ(param_.paddings.size(), 4UL);
  return true;
}

bool Pad2dOpLite::InferShape() const {
  if (!last_input_shapes.empty()) {
    if (last_input_shapes[0] == param_.X->dims() &&
        last_input_lods[0] == param_.X->lod()) {
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
  last_input_shapes.push_back(param_.X->dims());
  last_input_lods.push_back(param_.X->lod());

  if (!last_output_shapes.empty()) {
    last_output_shapes.clear();
    last_output_lods.clear();
  }
  last_output_shapes.push_back(param_.output->dims());
  last_output_lods.push_back(param_.output->lod());

  return true;
}

bool Pad2dOpLite::InferShape() const {
  // nchw
  auto x_dims = param_.X->dims();
  int out_h = x_dims[2] + param_.paddings[0] + param_.paddings[1];
  int out_w = x_dims[3] + param_.paddings[2] + param_.paddings[3];
  param_.Out->Resize(lite::DDim({x_dims[0], x_dims[1], out_h, out_w}));
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool Pad2dOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  param_.X = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.Out =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();
  param_.mode = op_desc.GetAttr<std::string>("mode");
  param_.pad_value = op_desc.GetAttr<float>("pad_value");
  if (op_desc.HasAttr("variable_padding") &&
      op_desc.GetAttr<bool>("variable_paddings")) {
    auto Paddings =
        scope->FindVar(op_desc.Input("Paddings").front())->GetMutable<Tensor>();
    auto ptr = Paddings->data<int>();
    if (Paddings->dims().size() < 4) {
      printf("Paddings size must be four: %d \n",
             static_cast<int>(Paddings->dims().size()));
      return false;
    }
    param_.paddings = {ptr[0], ptr[1], ptr[2], ptr[3]};
  } else {
    param_.paddings = op_desc.GetAttr<std::vector<int>>("paddings");
  }
  param_.data_format = op_desc.GetAttr<std::string>("data_format");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(pad2d, paddle::lite::operators::Pad2dOpLite);
