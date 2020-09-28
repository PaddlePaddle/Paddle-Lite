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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

// fixme: yolo box has updated, check arm kernel to get more info
int YoloBoxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindTensor(x_name);

  auto img_size_name = op_info->Input("ImgSize").front();
  auto img_size = scope->FindTensor(img_size_name);

  auto boxes_name = op_info->Output("Boxes").front();
  auto scores_name = op_info->Output("Scores").front();

  auto anchors = op_info->GetAttr<std::vector<int>>("anchors");
  auto class_num = op_info->GetAttr<int>("class_num");
  auto conf_thresh = op_info->GetAttr<float>("conf_thresh");
  auto downsample_ratio = op_info->GetAttr<int>("downsample_ratio");

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // ImgSize node
  std::shared_ptr<Node> img_size_node = nullptr;
  if (graph->Has(img_size_name)) {
    img_size_node = graph->Get(img_size_name);
  } else {
    img_size_node = graph->Add(img_size_name, *img_size);
  }

  // Softmax node
  auto yolo_box_data =
      graph->builder_.CreateYoloBox(*x_node->data(),
                                    *img_size_node->data(),
                                    CvtShape<xtcl::Integer>(anchors),
                                    class_num,
                                    conf_thresh,
                                    downsample_ratio);
  graph->Add(boxes_name, graph->builder_.GetField(yolo_box_data, 0));
  graph->Add(scores_name, graph->builder_.GetField(yolo_box_data, 1));

  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(yolo_box,
                         kXPU,
                         paddle::lite::subgraph::xpu::YoloBoxConverter);
