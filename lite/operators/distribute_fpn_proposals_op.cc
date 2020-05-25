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

#include "lite/operators/distribute_fpn_proposals_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool DistributeFpnProposalsOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.fpn_rois);
  CHECK_OR_FALSE(param_.restore_index);
  CHECK_OR_FALSE(param_.multi_fpn_rois.size() > 1);
  CHECK_OR_FALSE(param_.max_level >= param_.min_level);
  size_t num_out_rois =
      static_cast<size_t>(param_.max_level - param_.min_level + 1);
  CHECK_OR_FALSE(num_out_rois == param_.multi_fpn_rois.size());
  return true;
}

bool DistributeFpnProposalsOpLite::InferShapeImpl() const {
  int num_out_rois = param_.max_level - param_.min_level + 1;
  for (int i = 0; i < num_out_rois; i++) {
    param_.multi_fpn_rois[i]->Resize({-1, 4});
  }
  param_.restore_index->Resize({-1, 1});
  return true;
}

bool DistributeFpnProposalsOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                              lite::Scope *scope) {
  auto fpn_rois = op_desc.Input("FpnRois").front();
  param_.fpn_rois = scope->FindVar(fpn_rois)->GetMutable<lite::Tensor>();

  auto multi_fpn_rois = op_desc.Output("MultiFpnRois");
  for (const auto &name : multi_fpn_rois) {
    param_.multi_fpn_rois.push_back(
        scope->FindVar(name)->GetMutable<lite::Tensor>());
  }
  auto restore_index = op_desc.Output("RestoreIndex").front();
  param_.restore_index =
      scope->FindVar(restore_index)->GetMutable<lite::Tensor>();
  param_.min_level = op_desc.GetAttr<int>("min_level");
  param_.max_level = op_desc.GetAttr<int>("max_level");
  param_.refer_level = op_desc.GetAttr<int>("refer_level");
  param_.refer_scale = op_desc.GetAttr<int>("refer_scale");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(distribute_fpn_proposals,
                 paddle::lite::operators::DistributeFpnProposalsOpLite);
