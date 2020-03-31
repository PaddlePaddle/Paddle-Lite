// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/lookup_table_dequant_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool LookupTableDequantOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.W)
  CHECK_OR_FALSE(param_.Ids)
  CHECK_OR_FALSE(param_.Out)

  const auto& table_dims = param_.W->dims();
  const auto& ids_dims = param_.Ids->dims();

  int ids_rank = ids_dims.size();

  CHECK_EQ_OR_FALSE(table_dims.size(), 2);
  CHECK_EQ_OR_FALSE(ids_dims[ids_rank - 1], 1);
  CHECK_GT_OR_FALSE(table_dims[1], 2);
  return true;
}

bool LookupTableDequantOpLite::InferShapeImpl() const {
  const auto& table_dims = param_.W->dims();
  const auto& ids_dims = param_.Ids->dims();

  auto out_dims = ids_dims;
  int ids_rank = ids_dims.size();
  out_dims[ids_rank - 1] = (table_dims[1] - 2) * 4;

  param_.Out->Resize(out_dims);
  param_.Out->set_lod(param_.Ids->lod());
  return true;
}

bool LookupTableDequantOpLite::AttachImpl(const cpp::OpDesc& op_desc,
                                          lite::Scope* scope) {
  auto input = op_desc.Input("W").front();
  auto ids = op_desc.Input("Ids").front();
  auto out = op_desc.Output("Out").front();

  param_.W = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.Ids = scope->FindVar(ids)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();

  param_.padding_idx = op_desc.GetAttr<int64_t>("padding_idx");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(lookup_table_dequant,
                 paddle::lite::operators::LookupTableDequantOpLite)
