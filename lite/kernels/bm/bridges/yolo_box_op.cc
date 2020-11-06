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
#include <iostream>
#include <string>
#include <vector>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

// fixme: yolo box has updated, check arm kernel to get more info
int YoloBoxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);

  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  auto img_size_var_name = op_info->Input("ImgSize").front();
  auto img_size = scope->FindVar(img_size_var_name)->GetMutable<lite::Tensor>();
  auto img_size_dims = img_size->dims();
  auto boxes_var_name = op_info->Output("Boxes").front();
  auto boxes = scope->FindVar(boxes_var_name)->GetMutable<lite::Tensor>();
  auto boxes_dims = boxes->dims();
  auto scores_var_name = op_info->Output("Scores").front();
  auto scores = scope->FindVar(scores_var_name)->GetMutable<lite::Tensor>();
  auto scores_dims = scores->dims();
  std::vector<int32_t> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int32_t>(x_dims[i]);
  }
  std::vector<int32_t> i_img_size_shape_data(img_size_dims.size());
  for (size_t i = 0; i < img_size_dims.size(); i++) {
    i_img_size_shape_data[i] = static_cast<int32_t>(img_size_dims[i]);
  }
  std::vector<int32_t> i_boxes_shape_data(boxes_dims.size());
  for (size_t i = 0; i < boxes_dims.size(); i++) {
    i_boxes_shape_data[i] = static_cast<int32_t>(boxes_dims[i]);
  }
  std::vector<int32_t> i_scores_shape_data(scores_dims.size());
  for (size_t i = 0; i < scores_dims.size(); i++) {
    i_scores_shape_data[i] = static_cast<int32_t>(scores_dims[i]);
  }

  auto class_num = op_info->GetAttr<int>("class_num");
  auto downsample_ratio = op_info->GetAttr<int>("downsample_ratio");
  auto conf_thresh = op_info->GetAttr<float>("conf_thresh");
  auto anchors = op_info->GetAttr<std::vector<int>>("anchors");
  CHECK_LE(anchors.size(), 100);
  user_cpu_param_t bm_param;
  bm_param.op_type = USER_PADDLE_YOLO_BOX;
  bm_param.u.yolo_box_param.class_num = class_num;
  bm_param.u.yolo_box_param.downsample_ratio = downsample_ratio;
  bm_param.u.yolo_box_param.conf_thresh = conf_thresh;
  memset(bm_param.u.yolo_box_param.anchors, 0, 100 * sizeof(int));
  memcpy(bm_param.u.yolo_box_param.anchors,
         &anchors[0],
         anchors.size() * sizeof(int));
  bm_param.u.yolo_box_param.anchors_size = anchors.size();
  int32_t input_num = 2;
  int32_t output_num = 2;
  int32_t* in_shape[2];
  int32_t in_dim[2];
  const char* in_name[2];
  in_shape[0] = &i_x_shape_data[0];
  in_shape[1] = &i_img_size_shape_data[0];
  in_dim[0] = x_dims.size();
  in_dim[1] = img_size_dims.size();
  in_name[0] = static_cast<const char*>(x_var_name.c_str());
  in_name[1] = static_cast<const char*>(img_size_var_name.c_str());
  int32_t* out_shape[2];
  int32_t out_dim[2];
  const char* out_name[2];
  out_shape[0] = &i_boxes_shape_data[0];
  out_shape[1] = &i_scores_shape_data[0];
  out_dim[0] = boxes_dims.size();
  out_dim[1] = scores_dims.size();
  out_name[0] = static_cast<const char*>(boxes_var_name.c_str());
  out_name[1] = static_cast<const char*>(scores_var_name.c_str());

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
  graph->AddNode(boxes_var_name);
  graph->AddNode(scores_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(yolo_box,
                         kBM,
                         paddle::lite::subgraph::bm::YoloBoxConverter);
