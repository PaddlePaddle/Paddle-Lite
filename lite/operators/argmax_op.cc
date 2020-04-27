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

#include "lite/operators/argmax_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool ArgmaxOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  CHECK_OR_FALSE(param_.Axis < static_cast<int>((param_.X)->dims().size()));
  CHECK_OR_FALSE(param_.Axis >= static_cast<int>(-(param_.X)->dims().size()));
  return true;
}

bool ArgmaxOpLite::InferShapeImpl() const {
  auto x_dims = param_.X->dims();
  int x_rank = x_dims.size();
  int axis = param_.Axis;
  if (axis < 0) {
    axis += x_rank;
  }

  std::vector<int64_t> out_dims;
  for (int64_t i = 0; i < axis; i++) out_dims.push_back(x_dims[i]);
  for (int64_t i = axis + 1; i < x_rank; i++) out_dims.push_back(x_dims[i]);

  // Set output dims
  param_.Out->Resize(lite::DDim(out_dims));
  return true;
}

// TODO(Superjomn) replace framework::OpDesc with a lite one.
bool ArgmaxOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto out = op_desc.Output("Out").front();

  param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.Axis = op_desc.GetAttr<int64_t>("axis");

  return true;
}

#ifdef LITE_WITH_OPS
float ArgmaxOpLite::GetGops(){
  InferShapeImpl();
  auto x_dims = param_.X->dims();
  auto out_dims = param_.Out->dims();
  auto axis = param_.Axis;
  int x_rank = x_dims.size();
  int numel = out_dims.production();
  int max_num = 1;
  if (axis < 0) {
    axis += x_rank;
  }
  for (int64_t i = axis + 1; i < x_rank; i++) max_num *= x_dims[i];
  float gops = 1.0f;
  for (int i = 1; i <= max_num; i++) gops *= i;
  return gops * numel;
}
#endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(arg_max, paddle::lite::operators::ArgmaxOpLite);
