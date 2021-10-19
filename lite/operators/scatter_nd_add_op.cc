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

#include "lite/operators/scatter_nd_add_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool ScatterNdAddOp::InferShapeImpl() const {
  auto index_dims = param_.indexs->dims();
  auto update_dims = param_.updates->dims();
  auto input_dims = param_.x->dims();
  auto index_dims_size = index_dims.size();
  auto update_dims_size = update_dims.size();
  auto input_dims_size = input_dims.size();
  CHECK_LE(index_dims[index_dims_size - 1], input_dims_size)
      << "Input(Index).shape[-1] should be no greater than Input(X).rank";
  CHECK_GE(input_dims_size, 2L)
      << "The rank of Input(Index) should be greater than 1";
  std::vector<int64_t> r_updates_dims;
  for (int64_t i = 0; i < index_dims_size - 1; ++i) {
    r_updates_dims.emplace_back(index_dims[i]);
  }
  for (int64_t i = index_dims[index_dims_size - 1]; i < input_dims_size; ++i) {
    r_updates_dims.emplace_back(input_dims[i]);
  }

  CHECK_EQ(r_updates_dims.size(), update_dims_size)
      << "Updates has wrong shape";

  for (int64_t i = 0; i < update_dims_size; ++i) {
    CHECK_EQ(r_updates_dims[i], update_dims[i]) << "Updates has wrong shape";
  }
  param_.output->Resize(input_dims);
  return true;
}

bool ScatterNdAddOp::AttachImpl(const cpp::OpDesc &op_desc,
                                lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto indexs = op_desc.Input("Index").front();
  auto updates = op_desc.Input("Updates").front();
  auto output = op_desc.Output("Out").front();

  param_.x = scope->FindVar(x)->GetMutable<Tensor>();
  param_.indexs = scope->FindVar(indexs)->GetMutable<Tensor>();
  param_.updates = scope->FindVar(updates)->GetMutable<Tensor>();
  param_.output = scope->FindMutableTensor(output);

  CHECK(param_.x);
  CHECK(param_.indexs);
  CHECK(param_.updates);
  CHECK(param_.output);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(scatter_nd_add, paddle::lite::operators::ScatterNdAddOp);
