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

#include "lite/operators/__xpu__mmdnn_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUMMDNNBidEmbGrnnAttOp::CheckShape() const { return true; }

bool XPUMMDNNBidEmbGrnnAttOp::InferShapeImpl() const {
  auto& id_dims = param_.id0->dims();
  auto& id_lod = param_.id0->lod()[0];
  auto& emb_tbl_dims = param_.emb_tbl->dims();
  auto& grnn_wh_dims = param_.grnn_rv_wh->dims();

  param_.grnn_fw_pool_out->Resize(
      {(int64_t)id_lod.size() - 1, grnn_wh_dims[2]});
  param_.grnn_rv_pool_out->Resize(
      {(int64_t)id_lod.size() - 1, grnn_wh_dims[2]});
  param_.att_pool_out->Resize(
      {(int64_t)id_lod.size() - 1, 2 * grnn_wh_dims[2]});
  param_.concat_3in1_out->Resize({id_dims[0], 3 * grnn_wh_dims[2]});
  param_.concat_3in1_out->set_lod({id_lod});
  param_.emb_fw_out->Resize({id_dims[0], emb_tbl_dims[1]});
  param_.emb_fw_out->set_lod({id_lod});
  return true;
}

bool XPUMMDNNBidEmbGrnnAttOp::AttachImpl(const cpp::OpDesc& op_desc,
                                         lite::Scope* scope) {
  param_.id0 =
      scope->FindVar(op_desc.Input("id0").front())->GetMutable<lite::Tensor>();
  param_.id1 =
      scope->FindVar(op_desc.Input("id1").front())->GetMutable<lite::Tensor>();
  param_.emb_tbl = scope->FindVar(op_desc.Input("emb_tbl").front())
                       ->GetMutable<lite::Tensor>();
  param_.grnn_fw_wh = scope->FindVar(op_desc.Input("grnn_fw_wh").front())
                          ->GetMutable<lite::Tensor>();
  param_.grnn_fw_wi = scope->FindVar(op_desc.Input("grnn_fw_wi").front())
                          ->GetMutable<lite::Tensor>();
  param_.grnn_rv_wh = scope->FindVar(op_desc.Input("grnn_rv_wh").front())
                          ->GetMutable<lite::Tensor>();
  param_.grnn_rv_wi = scope->FindVar(op_desc.Input("grnn_rv_wi").front())
                          ->GetMutable<lite::Tensor>();
  param_.att_fc_w = scope->FindVar(op_desc.Input("att_fc_w").front())
                        ->GetMutable<lite::Tensor>();
  param_.att_fc_b = scope->FindVar(op_desc.Input("att_fc_b").front())
                        ->GetMutable<lite::Tensor>();

  param_.grnn_fw_pool_out =
      scope->FindVar(op_desc.Output("grnn_fw_pool_out").front())
          ->GetMutable<lite::Tensor>();
  param_.grnn_rv_pool_out =
      scope->FindVar(op_desc.Output("grnn_rv_pool_out").front())
          ->GetMutable<lite::Tensor>();
  param_.att_pool_out = scope->FindVar(op_desc.Output("att_pool_out").front())
                            ->GetMutable<lite::Tensor>();
  param_.concat_3in1_out =
      scope->FindVar(op_desc.Output("concat_3in1_out").front())
          ->GetMutable<lite::Tensor>();
  param_.emb_fw_out = scope->FindVar(op_desc.Output("emb_fw_out").front())
                          ->GetMutable<lite::Tensor>();

  param_.grnn_fw_wh_maxs =
      op_desc.GetAttr<std::vector<float>>("grnn_fw_wh_maxs");
  param_.grnn_fw_wi_maxs =
      op_desc.GetAttr<std::vector<float>>("grnn_fw_wi_maxs");
  param_.grnn_rv_wh_maxs =
      op_desc.GetAttr<std::vector<float>>("grnn_rv_wh_maxs");
  param_.grnn_rv_wi_maxs =
      op_desc.GetAttr<std::vector<float>>("grnn_rv_wi_maxs");
  param_.att_fc_w_max = op_desc.GetAttr<float>("att_fc_w_max");
  return true;
}

bool XPUMMDNNBidEmbAttOp::CheckShape() const { return true; }

bool XPUMMDNNBidEmbAttOp::InferShapeImpl() const {
  auto& id_dims = param_.id0->dims();
  auto& id_lod = param_.id0->lod()[0];
  auto& emb_tbl_dims = param_.emb_tbl->dims();

  param_.att_pool_out->Resize({(int64_t)id_lod.size() - 1, emb_tbl_dims[1]});
  param_.emb_fw_out->Resize({id_dims[0], emb_tbl_dims[1]});
  param_.emb_fw_out->set_lod({id_lod});
  return true;
}

bool XPUMMDNNBidEmbAttOp::AttachImpl(const cpp::OpDesc& op_desc,
                                     lite::Scope* scope) {
  param_.id0 =
      scope->FindVar(op_desc.Input("id0").front())->GetMutable<lite::Tensor>();
  param_.id1 =
      scope->FindVar(op_desc.Input("id1").front())->GetMutable<lite::Tensor>();
  param_.emb_tbl = scope->FindVar(op_desc.Input("emb_tbl").front())
                       ->GetMutable<lite::Tensor>();
  param_.att_fc_w = scope->FindVar(op_desc.Input("att_fc_w").front())
                        ->GetMutable<lite::Tensor>();
  param_.att_fc_b = scope->FindVar(op_desc.Input("att_fc_b").front())
                        ->GetMutable<lite::Tensor>();

  param_.att_pool_out = scope->FindVar(op_desc.Output("att_pool_out").front())
                            ->GetMutable<lite::Tensor>();
  param_.emb_fw_out = scope->FindVar(op_desc.Output("emb_fw_out").front())
                          ->GetMutable<lite::Tensor>();

  param_.att_fc_w_max = op_desc.GetAttr<float>("att_fc_w_max");
  return true;
}

bool XPUMMDNNMatchConvTopkOp::CheckShape() const { return true; }

bool XPUMMDNNMatchConvTopkOp::InferShapeImpl() const {
  int channel_num = param_.channel_num;
  std::vector<int> topks = param_.topks;
  auto row_dim = param_.input_x->dims();
  auto num_k = topks.size();
  auto row_shape_0 = row_dim[0];
  std::vector<int64_t> vec_out_shape;
  vec_out_shape.push_back(row_shape_0);
  vec_out_shape.push_back(channel_num * num_k);

  param_.topk_out->Resize(lite::DDim(vec_out_shape));
  param_.topk_out->set_lod(param_.input_x->lod());
  return true;
}

bool XPUMMDNNMatchConvTopkOp::AttachImpl(const cpp::OpDesc& op_desc,
                                         lite::Scope* scope) {
  param_.input_x = scope->FindVar(op_desc.Input("input_x").front())
                       ->GetMutable<lite::Tensor>();
  param_.input_y = scope->FindVar(op_desc.Input("input_y").front())
                       ->GetMutable<lite::Tensor>();
  param_.input_w = scope->FindVar(op_desc.Input("input_w").front())
                       ->GetMutable<lite::Tensor>();
  param_.conv_w = scope->FindVar(op_desc.Input("conv_w").front())
                      ->GetMutable<lite::Tensor>();

  param_.topk_out = scope->FindVar(op_desc.Output("topk_out").front())
                        ->GetMutable<lite::Tensor>();

  param_.input_w_max = op_desc.GetAttr<float>("input_w_max");
  param_.conv_w_max = op_desc.GetAttr<float>("conv_w_max");
  param_.topks = op_desc.GetAttr<std::vector<int>>("topks");
  param_.channel_num = op_desc.GetAttr<int>("channel_num");
  param_.dim_t = op_desc.GetAttr<int>("dim_t");
  return true;
}

bool XPUMMDNNMergeAllOp::CheckShape() const { return true; }

bool XPUMMDNNMergeAllOp::InferShapeImpl() const {
  int64_t dim0 = param_.concat_7in1_x[0]->dims()[0];
  int64_t dim1 = param_.fc2_w->dims()[0];
  std::vector<int64_t> vec_out_shape;
  vec_out_shape.push_back(dim0);
  vec_out_shape.push_back(dim1);

  param_.out->Resize(lite::DDim(vec_out_shape));
  return true;
}

bool XPUMMDNNMergeAllOp::AttachImpl(const cpp::OpDesc& op_desc,
                                    lite::Scope* scope) {
  param_.concat_7in1_x.clear();
  for (auto& name : op_desc.Input("concat_7in1_x")) {
    auto t = scope->FindVar(name)->GetMutable<lite::Tensor>();
    param_.concat_7in1_x.push_back(t);
  }
  param_.concat_2in1_x.clear();
  for (auto& name : op_desc.Input("concat_2in1_x")) {
    auto t = scope->FindVar(name)->GetMutable<lite::Tensor>();
    param_.concat_2in1_x.push_back(t);
  }
  param_.grnn_fw_wh = scope->FindVar(op_desc.Input("grnn_fw_wh").front())
                          ->GetMutable<lite::Tensor>();
  param_.grnn_fw_wi = scope->FindVar(op_desc.Input("grnn_fw_wi").front())
                          ->GetMutable<lite::Tensor>();
  param_.grnn_rv_wh = scope->FindVar(op_desc.Input("grnn_rv_wh").front())
                          ->GetMutable<lite::Tensor>();
  param_.grnn_rv_wi = scope->FindVar(op_desc.Input("grnn_rv_wi").front())
                          ->GetMutable<lite::Tensor>();
  param_.fc0_w = scope->FindVar(op_desc.Input("fc0_w").front())
                     ->GetMutable<lite::Tensor>();
  param_.fc0_b = scope->FindVar(op_desc.Input("fc0_b").front())
                     ->GetMutable<lite::Tensor>();
  param_.fc1_w = scope->FindVar(op_desc.Input("fc1_w").front())
                     ->GetMutable<lite::Tensor>();
  param_.fc1_b = scope->FindVar(op_desc.Input("fc1_b").front())
                     ->GetMutable<lite::Tensor>();
  param_.fc2_w = scope->FindVar(op_desc.Input("fc2_w").front())
                     ->GetMutable<lite::Tensor>();
  param_.fc2_b = scope->FindVar(op_desc.Input("fc2_b").front())
                     ->GetMutable<lite::Tensor>();

  param_.out =
      scope->FindVar(op_desc.Output("out").front())->GetMutable<lite::Tensor>();

  param_.grnn_fw_wh_maxs =
      op_desc.GetAttr<std::vector<float>>("grnn_fw_wh_maxs");
  param_.grnn_fw_wi_maxs =
      op_desc.GetAttr<std::vector<float>>("grnn_fw_wi_maxs");
  param_.grnn_rv_wh_maxs =
      op_desc.GetAttr<std::vector<float>>("grnn_rv_wh_maxs");
  param_.grnn_rv_wi_maxs =
      op_desc.GetAttr<std::vector<float>>("grnn_rv_wi_maxs");
  param_.fc0_w_max = op_desc.GetAttr<float>("fc0_w_max");
  param_.fc1_w_max = op_desc.GetAttr<float>("fc1_w_max");
  param_.fc2_w_max = op_desc.GetAttr<float>("fc2_w_max");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__mmdnn_bid_emb_grnn_att,
                 paddle::lite::operators::XPUMMDNNBidEmbGrnnAttOp);
REGISTER_LITE_OP(__xpu__mmdnn_bid_emb_att,
                 paddle::lite::operators::XPUMMDNNBidEmbAttOp);
REGISTER_LITE_OP(__xpu__mmdnn_match_conv_topk,
                 paddle::lite::operators::XPUMMDNNMatchConvTopkOp);
REGISTER_LITE_OP(__xpu__mmdnn_merge_all,
                 paddle::lite::operators::XPUMMDNNMergeAllOp);
