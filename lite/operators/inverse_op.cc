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

#include "lite/operators/inverse_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool InverseOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.Input);
  CHECK_OR_FALSE(param_.Output);
  auto input_dims = param_.Input->dims();

  int64_t input_rank = input_dims.size();
  CHECK_OR_FALSE(input_rank >= 2);
  if (input_dims[input_rank - 2] > 0 && input_dims[input_rank - 1] > 0)
    CHECK_OR_FALSE(input_dims[input_rank - 2] == input_dims[input_rank - 1]);
  return true;
}

bool InverseOpLite::InferShapeImpl() const {
  auto x_dims = param_.Input->dims();
  int x_rank = x_dims.size();
  std::vector<int64_t> out_dims;
  for (int64_t i = 0; i < x_rank; i++) out_dims.push_back(x_dims[i]);
  // Set output dims
  param_.Output->Resize(lite::DDim(out_dims));
  return true;
}

bool InverseOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto Input = op_desc.Input("Input").front();
  auto Output = op_desc.Output("Output").front();
  param_.Input = scope->FindVar(Input)->GetMutable<lite::Tensor>();
  param_.Output = scope->FindVar(Output)->GetMutable<lite::Tensor>();
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(inverse, paddle::lite::operators::InverseOpLite);
