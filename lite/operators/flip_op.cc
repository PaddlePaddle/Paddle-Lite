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

#include "lite/operators/flip_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FlipOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool FlipOp::InferShapeImpl() const {
  auto x_dims = param_.X->dims();
  auto flip_dims = param_.axis;
  auto flip_dims_size = flip_dims.size();
  CHECK_GT_OR_FALSE(flip_dims_size, 0);

  auto min_max_d = std::minmax_element(flip_dims.begin(), flip_dims.end());
  CHECK_LT(*min_max_d.first, static_cast<int32_t>(x_dims.size()))
      << "min(axes) should be less than the input tensor X's "
         "axes of FlipOp. But received min(axes) = "
      << *min_max_d.first << ",  X's axes = " << x_dims.size()
      << ", X's shape = [" << x_dims << "]";

  CHECK_GE(*min_max_d.first, static_cast<int32_t>(x_dims.size() * -1))
      << "min(axes) should be greater than the input tensor X's "
         "axes of FlipOp times -1. But received min(axes) = "
      << *min_max_d.first << ",  X's axes = " << x_dims.size()
      << ", X's shape = [" << x_dims << "]";

  CHECK_GE(*min_max_d.second, static_cast<int32_t>(x_dims.size() * -1))
      << "max(axes) should be greater than the input tensor X's "
         "axes of FlipOp times -1. But received max(axes) = "
      << *min_max_d.second << ",  X's axes = " << x_dims.size()
      << ", X's shape = [" << x_dims << "]";

  CHECK_LT(*min_max_d.second, static_cast<int32_t>(x_dims.size()))
      << "min(axes) should be less than the input tensor X's "
         "axes of FlipOp. But received min(axes) = "
      << *min_max_d.second << ",  X's axes = " << x_dims.size()
      << ", X's shape = [" << x_dims << "]";

  flip_dims.erase(std::unique(flip_dims.begin(), flip_dims.end()),
                  flip_dims.end());
  CHECK_EQ(flip_dims.size(), flip_dims_size)
      << "axes has duplicates, original flip axes size=" << flip_dims_size
      << ", but unique flip axes size=" << flip_dims.size() << ".";
  std::vector<int64_t> output_dims(x_dims.size());
  for (int i = 0; i < x_dims.size(); ++i) {
    output_dims[i] = x_dims[i];
  }

  param_.Out->Resize(output_dims);

  return true;
}

bool FlipOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto x_var = scope->FindVar(opdesc.Input("X").front());
  auto output_var = scope->FindVar(opdesc.Output("Out").front());
  CHECK(x_var);
  CHECK(output_var);
  param_.X = const_cast<lite::Tensor *>(&(x_var->Get<lite::Tensor>()));
  param_.Out = output_var->GetMutable<lite::Tensor>();
  param_.axis = opdesc.GetAttr<std::vector<int>>("axis");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(flip, paddle::lite::operators::FlipOp)
