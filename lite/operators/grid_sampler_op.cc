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
  CHECK(param_.x);
  CHECK(param_.grid);
  CHECK(param_.out);
  CHECK_EQ(param_.x->dims().size(), 4);
  CHECK_EQ(param_.grid->dims().size(), 4);
  return true;
}

bool GridSamplerOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  auto grid_dims = param_.grid->dims();
  auto out_dims = param_.out->dims();
  std::vector<int64_t> out_shape{
      x_dims[0], x_dims[1], grid_dims[1], grid_dims[2]};
  param_.out->Resize(out_shape);
  return true;
}

bool GridSamplerOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  param_.x = scope->FindTensor(op_desc.Input("X").front());
  param_.grid = scope->FindTensor(op_desc.Input("Grid").front());
  param_.out = scope->FindMutableTensor(op_desc.Output("Output").front());

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
