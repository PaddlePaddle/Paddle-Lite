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

#include "lite/operators/generate_proposals_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool GenerateProposalsOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.Scores);
  CHECK_OR_FALSE(param_.BboxDeltas);
  CHECK_OR_FALSE(param_.ImInfo);
  CHECK_OR_FALSE(param_.Anchors);
  CHECK_OR_FALSE(param_.Variances);
  CHECK_OR_FALSE(param_.RpnRois);
  CHECK_OR_FALSE(param_.RpnRoiProbs);

  auto scores_dims = param_.Scores->dims();
  auto bbox_dims = param_.BboxDeltas->dims();
  auto im_info_dims = param_.ImInfo->dims();
  auto anchors_dims = param_.Anchors->dims();
  auto vars_dims = param_.Variances->dims();

  CHECK_OR_FALSE(bbox_dims[1] = 4 * scores_dims[1]);
  CHECK_OR_FALSE(scores_dims[1] == anchors_dims[2]);
  CHECK_OR_FALSE(anchors_dims == vars_dims);

  return true;
}

bool GenerateProposalsOpLite::InferShapeImpl() const {
  param_.RpnRois->Resize(std::vector<int64_t>({-1, 4}));
  param_.RpnRoiProbs->Resize(std::vector<int64_t>({-1, 1}));
  return true;
}

bool GenerateProposalsOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                         lite::Scope *scope) {
  // inputs
  param_.Scores = scope->FindVar(op_desc.Input("Scores").front())
                      ->GetMutable<lite::Tensor>();
  param_.BboxDeltas = scope->FindVar(op_desc.Input("BboxDeltas").front())
                          ->GetMutable<lite::Tensor>();
  param_.ImInfo = scope->FindVar(op_desc.Input("ImInfo").front())
                      ->GetMutable<lite::Tensor>();
  param_.Anchors = scope->FindVar(op_desc.Input("Anchors").front())
                       ->GetMutable<lite::Tensor>();
  param_.Variances = scope->FindVar(op_desc.Input("Variances").front())
                         ->GetMutable<lite::Tensor>();

  // attrs
  param_.pre_nms_topN = op_desc.GetAttr<int>("pre_nms_topN");
  param_.post_nms_topN = op_desc.GetAttr<int>("post_nms_topN");
  param_.nms_thresh = op_desc.GetAttr<float>("nms_thresh");
  param_.min_size = op_desc.GetAttr<float>("min_size");
  param_.eta = op_desc.GetAttr<float>("eta");

  // outs
  param_.RpnRois = scope->FindVar(op_desc.Output("RpnRois").front())
                       ->GetMutable<lite::Tensor>();
  param_.RpnRoiProbs = scope->FindVar(op_desc.Output("RpnRoiProbs").front())
                           ->GetMutable<lite::Tensor>();
  if (op_desc.HasOutput("RpnRoisLod") &&
      !op_desc.Output("RpnRoisLod").empty()) {
    param_.RpnRoisLod = scope->FindVar(op_desc.Output("RpnRoisLod").front())
                            ->GetMutable<lite::Tensor>();
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(generate_proposals,
                 paddle::lite::operators::GenerateProposalsOpLite);
