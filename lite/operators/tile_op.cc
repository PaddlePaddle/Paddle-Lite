// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/tile_op.h"
#include <algorithm>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool TileOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  return true;
}

bool TileOp::InferShapeImpl() const {
  const auto &out = param_.Out;
  const auto x_dims = param_.X->dims();
  std::vector<int> repeat_times;
  if (param_.RepeatTimes) {
    auto repeat_times_size = param_.RepeatTimes->data_size();
    for (int64_t i = 0; i < repeat_times_size; i++) {
      repeat_times.push_back(param_.RepeatTimes->data<int>()[i]);
    }
  } else if (param_.repeat_times_tensor.size() != 0) {
    auto repeat_times_size = param_.repeat_times_tensor.size();
    for (int64_t i = 0; i < repeat_times_size; i++) {
      auto temp = param_.repeat_times_tensor[i];
      repeat_times.push_back(*temp->data<int>());
    }
  } else {
    repeat_times = param_.repeat_times;
  }
  param_.repeat_times = repeat_times;
  if (repeat_times.size() == 0) {
    repeat_times = std::vector<int>(x_dims.size(), -1);
  }
  CHECK_GE(x_dims.size(), 1)
      << "The rank of the input 'x' for tile op "
      << "must be positive integers, but the value received is "
      << x_dims.size();

  CHECK_LE(x_dims.size(), 6)
      << "The rank of the input 'x' for tile op "
      << "must not be greater than 6, but the value received is "
      << x_dims.size();

  CHECK_LE(repeat_times.size(), 6)
      << "The size of the shape of input 'repeat_times' for tile op "
      << "must not be greater than 6, but the value received is "
      << repeat_times.size();

  CHECK_GE(repeat_times.size(), 1)
      << "The size of the shape of input 'repeat_times' for tile op "
      << "must be positive integers, but the value received is "
      << repeat_times.size();

  auto out_rank =
      std::max(static_cast<size_t>(x_dims.size()), repeat_times.size());

  std::vector<int64_t> out_shape(out_rank);
  auto x_dim_vec = x_dims.Vectorize();
  if (x_dim_vec.size() > repeat_times.size()) {
    auto diff = x_dim_vec.size() - repeat_times.size();
    repeat_times.insert(repeat_times.begin(), diff, -1);
  } else {
    auto diff = repeat_times.size() - x_dim_vec.size();
    x_dim_vec.insert(x_dim_vec.begin(), diff, -1);
  }
  for (size_t i = 0; i < repeat_times.size(); ++i) {
    if (x_dim_vec[i] == -1 || repeat_times[i] == -1) {
      out_shape[i] = -1;
    } else {
      CHECK_GT(repeat_times[i], 0)
          << "Every element of the input 'repeat_times' for tile op must be "
          << "greater than 1, but the value given is ",
          repeat_times[i];
      out_shape[i] = x_dim_vec[i] * repeat_times[i];
    }
  }
  out->Resize(out_shape);
  if (out_shape[0] == x_dims[0]) {
    param_.X->set_lod(param_.Out->lod());
  }
  return true;
}

bool TileOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = scope->FindMutableTensor(opdesc.Input("X").front());
  if (opdesc.HasInput("RepeatTimes") && !opdesc.Input("RepeatTimes").empty()) {
    param_.RepeatTimes =
        scope->FindMutableTensor(opdesc.Input("RepeatTimes").front());
  } else if (opdesc.HasInput("repeat_times_tensor") &&
             (opdesc.Input("repeat_times_tensor").size() != 0)) {
    auto temp = opdesc.Input("repeat_times_tensor");
    param_.repeat_times_tensor.clear();
    for (auto var : temp) {
      param_.repeat_times_tensor.push_back(
          scope->FindVar(var)->GetMutable<lite::Tensor>());
    }
  } else if (opdesc.HasAttr("repeat_times")) {
    param_.repeat_times = opdesc.GetAttr<std::vector<int>>("repeat_times");
  }
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(tile, paddle::lite::operators::TileOp);
