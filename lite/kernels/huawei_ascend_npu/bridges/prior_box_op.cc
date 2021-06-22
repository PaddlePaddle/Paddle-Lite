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

#include "lite/operators/prior_box_op.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int PriorBoxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input = scope->FindMutableTensor(input_name);
  auto image_name = op_info->Input("Image").front();
  auto image = scope->FindMutableTensor(image_name);
  auto boxes_name = op_info->Output("Boxes").front();
  // auto boxes = scope->FindMutableTensor(boxes_name);
  // auto variances_name = op_info->Output("Variances").front();
  // auto variances = scope->FindMutableTensor(variances_name);

  auto min_sizes = op_info->GetAttr<std::vector<float>>("min_sizes");
  auto max_sizes = op_info->GetAttr<std::vector<float>>("max_sizes");
  auto aspect_ratios = op_info->GetAttr<std::vector<float>>("aspect_ratios");
  auto variances = op_info->GetAttr<std::vector<float>>("variances");
  auto clip = op_info->GetAttr<bool>("clip");
  bool flip = op_info->HasAttr("flip") && op_info->GetAttr<bool>("flip");
  int img_w = op_info->HasAttr("img_w") ? op_info->GetAttr<int>("img_w") : 0;
  int img_h = op_info->HasAttr("img_h") ? op_info->GetAttr<int>("img_h") : 0;
  float step_w =
      op_info->HasAttr("step_w") ? op_info->GetAttr<float>("step_w") : 0;
  float step_h =
      op_info->HasAttr("step_h") ? op_info->GetAttr<float>("step_h") : 0;
  float offset =
      op_info->HasAttr("offset") ? op_info->GetAttr<float>("offset") : 0;

  // input node
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    input_node = graph->Add(input_name, *input);
  }

  // image node
  std::shared_ptr<Node> image_node = nullptr;
  if (graph->Has(image_name)) {
    image_node = graph->Get(image_name);
  } else {
    image_node = graph->Add(image_name, *image);
  }

  // priorBox node
  auto prior_box_node = graph->Add<ge::op::PriorBox>(boxes_name);
  auto prior_box_op = prior_box_node->data<ge::op::PriorBox>();
  prior_box_op->set_input_x(*input_node->data());
  prior_box_op->set_input_img(*image_node->data());

  prior_box_op->set_attr_min_size(min_sizes);
  prior_box_op->set_attr_max_size(max_sizes);
  prior_box_op->set_attr_aspect_ratio(aspect_ratios);
  prior_box_op->set_attr_flip(flip);
  prior_box_op->set_attr_clip(clip);
  prior_box_op->set_attr_variance(variances);
  prior_box_op->set_attr_step_h(step_h);
  prior_box_op->set_attr_step_w(step_w);
  prior_box_op->set_attr_offset(offset);
  prior_box_op->set_attr_img_h(img_h);
  prior_box_op->set_attr_img_w(img_w);

  INPUT_UPDATE(prior_box_op, x, input_node);
  INPUT_UPDATE(prior_box_op, img, image_node);
  OUTPUT_UPDATE(prior_box_op, y, prior_box_node);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    prior_box,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::PriorBoxConverter);
