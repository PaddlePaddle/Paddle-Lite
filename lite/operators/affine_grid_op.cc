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

#include "lite/operators/affine_grid_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool AffineGridOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);

  const auto x_dims = param_.X->dims();

  CHECK_OR_FALSE(x_dims.size() == 3);
  CHECK_OR_FALSE(x_dims[1] == 2 && x_dims[2] == 3);

  if (param_.output_shape.size() != 0) {
    CHECK_OR_FALSE(param_.output_shape.size() == 4);
  }
  return true;
}

bool AffineGridOpLite::InferShapeImpl() const {
  int N = param_.X->dims()[0];
  int H, W;
  if (param_.output_shape.size() == 0) {
    const auto out_shape = param_.OutputShape->data<int>();
    H = out_shape[2];
    W = out_shape[3];

  } else {
    H = param_.output_shape[2];
    W = param_.output_shape[3];
  }
  param_.Out->Resize(std::vector<int64_t>({N, H, W, 2}));

  return true;
}

bool AffineGridOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                  lite::Scope *scope) {
  auto x = op_desc.Input("Theta").front();
  auto output = op_desc.Output("Output").front();

  param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.output_shape = op_desc.GetAttr<std::vector<int>>("output_shape");
  if (param_.output_shape.size() == 0) {
    if (op_desc.HasInput("OutputShape")) {
      auto out_shape = op_desc.Input("OutputShape").front();
      param_.OutputShape =
          scope->FindVar(out_shape)->GetMutable<lite::Tensor>();
    } else {
      LOG(FATAL) << "The input 'OutputShape' of affine_grid Op should not be "
                    "null if 'output_shape' is not configured.";
    }
  }
  if (op_desc.HasAttr("align_corners")) {
    param_.align_corners = op_desc.GetAttr<bool>("align_corners");
  }

  param_.Out = scope->FindVar(output)->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(affine_grid, paddle::lite::operators::AffineGridOpLite);
