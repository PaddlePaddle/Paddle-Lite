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

#include "lite/operators/grid_sampler_op.h"
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace operators {

bool GridSamplerOp::CheckShape() const {
  CHECK_OR_FALSE(param_.x);
  CHECK_OR_FALSE(param_.out);
  CHECK_OR_FALSE(param_.grid);
#ifdef LITE_WITH_XPU
  return true;
#endif
  auto x_dims = param_.x->dims();
  auto grid_dims = param_.grid->dims();

  CHECK_EQ(x_dims.size(), 4UL) << "Input must have 4 dimensions.";
  CHECK_EQ(grid_dims.size(), 4UL) << "Grid must have 4 dimensions.";
  CHECK_EQ(grid_dims[0], x_dims[0])
      << "Input(X) dims[0] and Input(Grid) dims[0] should be equal.";
  CHECK_EQ(grid_dims[1], x_dims[2])
      << "Input(X) dims[2] and Input(Grid) dims[1] should be equal.";
  CHECK_EQ(grid_dims[2], x_dims[3])
      << "Input(X) dims[3] and Input(Grid) dims[2] should be equal.";

  return true;
}

bool GridSamplerOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  param_.out->Resize(x_dims);
  return true;
}

bool GridSamplerOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.x = scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.grid =
      scope->FindVar(op_desc.Input("Grid").front())->GetMutable<Tensor>();
  param_.out =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();
  param_.align_corners =
      scope->FindVar(op_desc.Output("Output").front())->GetMutable<Tensor>();

  if (op_desc.HasAttr("align_corners")) {
    param_.align_corners = op_desc.GetAttr<bool>("align_corners");
  }
  if (op_desc.HasAttr("padding_mode")) {
    param_.padding_mode = op_desc.GetAttr<std::string>("padding_mode");
  }
  if (op_desc.HasAttr("mode")) {
    param_.mode = op_desc.GetAttr<std::string>("mode");
  }

  return true;
}

} /* namespace operators */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_OP(grid_sampler, paddle::lite::operators::GridSamplerOp);
