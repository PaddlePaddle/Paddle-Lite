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

#include "lite/operators/sequence_topk_avg_pooling_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequenceTopkAvgPoolingOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.ROW);
  CHECK_OR_FALSE(param_.COLUMN);
  CHECK_OR_FALSE(param_.Out);
  CHECK_OR_FALSE(param_.pos);
  return true;
}

bool SequenceTopkAvgPoolingOpLite::InferShapeImpl() const {
  int channel_num = param_.channel_num;
  std::vector<int> topks = param_.topks;
  auto row_dim = param_.ROW->dims();
  auto num_k = topks.size();
  auto row_shape_0 = row_dim[0];
  std::vector<int64_t> vec_out_shape;
  vec_out_shape.push_back(row_shape_0);
  vec_out_shape.push_back(channel_num * num_k);

  param_.Out->Resize(lite::DDim(vec_out_shape));
  param_.Out->set_lod(param_.ROW->lod());
  return true;
}

bool SequenceTopkAvgPoolingOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                              lite::Scope *scope) {
  auto X = op_desc.Input("X").front();
  auto ROW = op_desc.Input("ROW").front();
  auto COLUMN = op_desc.Input("COLUMN").front();
  auto Out = op_desc.Output("Out").front();
  auto pos = op_desc.Output("pos").front();

  param_.X = scope->FindVar(X)->GetMutable<lite::Tensor>();
  param_.ROW = scope->FindVar(ROW)->GetMutable<lite::Tensor>();
  param_.COLUMN = scope->FindVar(COLUMN)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(Out)->GetMutable<lite::Tensor>();
  param_.pos = scope->FindVar(pos)->GetMutable<lite::Tensor>();
  param_.channel_num = op_desc.GetAttr<int>("channel_num");
  param_.topks = op_desc.GetAttr<std::vector<int>>("topks");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_topk_avg_pooling,
                 paddle::lite::operators::SequenceTopkAvgPoolingOpLite);
