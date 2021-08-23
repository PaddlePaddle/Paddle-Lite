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

#include <bmcompiler_if.h>
#include <user_bmcpu_common.h>
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

int MultiClassNMSConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto boxes_var_name = op_info->Input("BBoxes").front();
  auto boxes = scope->FindVar(boxes_var_name)->GetMutable<lite::Tensor>();
  auto boxes_dims = boxes->dims();
  std::vector<int32_t> i_boxes_shape_data(boxes_dims.size());
  for (size_t i = 0; i < boxes_dims.size(); i++) {
    i_boxes_shape_data[i] = static_cast<int32_t>(boxes_dims[i]);
  }
  auto score_var_name = op_info->Input("Scores").front();
  auto score = scope->FindVar(score_var_name)->GetMutable<lite::Tensor>();
  auto score_dims = score->dims();
  std::vector<int32_t> i_score_shape_data(score_dims.size());
  for (size_t i = 0; i < score_dims.size(); i++) {
    i_score_shape_data[i] = static_cast<int32_t>(score_dims[i]);
  }

  auto background_label = op_info->GetAttr<int>("background_label");
  auto keep_top_k = op_info->GetAttr<int>("keep_top_k");
  auto nms_top_k = op_info->GetAttr<int>("nms_top_k");
  auto score_threshold = op_info->GetAttr<float>("score_threshold");
  auto nms_threshold = op_info->GetAttr<float>("nms_threshold");
  auto nms_eta = op_info->GetAttr<float>("nms_eta");
  bool normalized = false;
  if (op_info->HasAttr("normalized")) {
    normalized = op_info->GetAttr<bool>("normalized");
  }

  auto out_var_name = op_info->Output("Out").front();
  auto out = scope->FindVar(out_var_name)->GetMutable<lite::Tensor>();
  std::vector<int64_t> vec_out_dim(score_dims.size());
  if (3 == score_dims.size()) {
    vec_out_dim[0] = score_dims[0];  // batch_size
    vec_out_dim[1] = keep_top_k;
    vec_out_dim[2] = 6;
  } else {
    vec_out_dim[0] = keep_top_k;
    vec_out_dim[1] = 6;
  }
  DDimLite out_dims(vec_out_dim);
  out->Resize(out_dims);
  out->mutable_data<float>();

  std::vector<int32_t> i_out_shape_data(out_dims.size());
  for (size_t i = 0; i < out_dims.size(); i++) {
    i_out_shape_data[i] = static_cast<int32_t>(out_dims[i]);
  }

  user_cpu_param_t bm_param;
  bm_param.op_type = USER_PADDLE_MULTICLASS_NMS;
  bm_param.u.multiclass_nms_param.background_label = background_label;
  bm_param.u.multiclass_nms_param.score_threshold = score_threshold;
  bm_param.u.multiclass_nms_param.keep_top_k = keep_top_k;
  bm_param.u.multiclass_nms_param.nms_top_k = nms_top_k;
  bm_param.u.multiclass_nms_param.nms_threshold = nms_threshold;
  bm_param.u.multiclass_nms_param.nms_eta = nms_eta;
  bm_param.u.multiclass_nms_param.normalized = normalized;

  int32_t input_num = 2;
  int32_t output_num = 1;
  int32_t* in_shape[2];
  int32_t in_dim[2];
  const char* in_name[2];
  in_shape[0] = &i_boxes_shape_data[0];
  in_shape[1] = &i_score_shape_data[0];
  in_dim[0] = boxes_dims.size();
  in_dim[1] = score_dims.size();
  in_name[0] = static_cast<const char*>(boxes_var_name.c_str());
  in_name[1] = static_cast<const char*>(score_var_name.c_str());
  int32_t* out_shape[2];
  int32_t out_dim[2];
  const char* out_name[2];
  out_shape[0] = &i_out_shape_data[0];
  out_dim[0] = out_dims.size();
  out_name[0] = static_cast<const char*>(out_var_name.c_str());

  std::vector<int64_t> vec_index_dim(score_dims.size());
  std::vector<int32_t> i_out_index_shape_data(score_dims.size());
  std::string out_index_name = "";
  if (op_type == "multiclass_nms2") {
    output_num = 2;
    out_index_name = op_info->Output("Index").front();
    auto out_index = scope->FindVar(out_index_name)->GetMutable<lite::Tensor>();
    if (3 == score_dims.size()) {
      vec_index_dim[0] = score_dims[0];
      vec_index_dim[1] = keep_top_k;
      vec_index_dim[2] = 1;
    } else {
      vec_index_dim[0] = keep_top_k;
      vec_index_dim[1] = 1;
    }
    DDimLite index_dims(vec_index_dim);
    out_index->Resize(index_dims);
    out_index->mutable_data<float>();
    for (size_t i = 0; i < index_dims.size(); i++) {
      i_out_index_shape_data[i] = static_cast<int32_t>(index_dims[i]);
    }
    out_shape[1] = &i_out_index_shape_data[0];
    out_dim[1] = index_dims.size();
    out_name[1] = static_cast<const char*>(out_index_name.c_str());
  }

  add_user_cpu_layer(graph->GetCompilerHandle(),
                     input_num,
                     in_shape,
                     in_dim,
                     in_name,
                     output_num,
                     out_shape,
                     out_dim,
                     out_name,
                     &bm_param,
                     static_cast<int>(sizeof(bm_param)));
  graph->AddNode(out_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(multiclass_nms,
                         kBM,
                         paddle::lite::subgraph::bm::MultiClassNMSConverter);
REGISTER_SUBGRAPH_BRIDGE(multiclass_nms2,
                         kBM,
                         paddle::lite::subgraph::bm::MultiClassNMSConverter);
