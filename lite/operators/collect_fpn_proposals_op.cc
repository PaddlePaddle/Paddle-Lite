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

#include "lite/operators/collect_fpn_proposals_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool CollectFpnProposalsOpLite::CheckShape() const {
  CHECK_OR_FALSE(!param_.multi_level_rois.empty());
  CHECK_OR_FALSE(!param_.multi_level_scores.empty());
  CHECK_OR_FALSE(param_.fpn_rois);

  for (auto item : param_.multi_level_rois) {
    auto dims = item->dims();
    CHECK_OR_FALSE(dims[1] == 4);
  }
  for (auto item : param_.multi_level_scores) {
    auto dims = item->dims();
    CHECK_OR_FALSE(dims[1] == 1);
  }
  for (int i = 0; i < param_.multi_level_rois.size(); i++) {
    auto roi = param_.multi_level_rois[i];
    auto roi_lod = roi->lod();
    auto score = param_.multi_level_scores[i];
    auto score_lod = score->lod();
    CHECK_OR_FALSE(roi_lod == score_lod);
  }
  return true;
}

bool CollectFpnProposalsOpLite::InferShapeImpl() const {
  param_.fpn_rois->Resize({param_.post_nms_topN, 4});
  if (param_.rois_num) param_.rois_num->Resize({-1});

  return true;
}

bool CollectFpnProposalsOpLite::AttachImpl(const cpp::OpDesc& op_desc,
                                           lite::Scope* scope) {
  auto rois_names = op_desc.Input("MultiLevelRois");
  param_.multi_level_rois.clear();
  for (const auto& var_name : rois_names) {
    param_.multi_level_rois.push_back(
        scope->FindVar(var_name)->GetMutable<lite::Tensor>());
  }
  auto scores_names = op_desc.Input("MultiLevelScores");
  param_.multi_level_scores.clear();
  for (const auto& var_name : scores_names) {
    param_.multi_level_scores.push_back(
        scope->FindVar(var_name)->GetMutable<lite::Tensor>());
  }
  if (op_desc.HasInput("MultiLevelRoIsNum")) {
    auto multi_rois_num = op_desc.Input("MultiLevelRoIsNum");
    param_.multi_rois_num.clear();
    for (const auto& name : multi_rois_num) {
      param_.multi_rois_num.push_back(
          scope->FindVar(name)->GetMutable<lite::Tensor>());
    }
  }

  auto fpn_rois = op_desc.Output("FpnRois").front();
  param_.fpn_rois = scope->FindVar(fpn_rois)->GetMutable<lite::Tensor>();

  if (op_desc.HasOutput("RoisNum")) {
    auto var = scope->FindVar(op_desc.Output("RoisNum").front());
    if (var != nullptr) {
      param_.rois_num = var->GetMutable<lite::Tensor>();
    }
  }
  param_.post_nms_topN = op_desc.GetAttr<int>("post_nms_topN");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(collect_fpn_proposals,
                 paddle::lite::operators::CollectFpnProposalsOpLite);
