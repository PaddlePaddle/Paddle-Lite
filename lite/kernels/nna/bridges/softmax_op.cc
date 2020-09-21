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

#include "lite/kernels/nna/bridges/graph.h"
#include "lite/kernels/nna/bridges/registry.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nna {

int SoftmaxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NNA] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto x_rank = x_dims.size();
  auto out_name = op_info->Output("Out").front();
  int axis = op_info->HasAttr("axis") ? op_info->GetAttr<int>("axis") : -1;
  if (axis < 0) {
    axis += x_rank;
  }

  // for quantization
  float output_scale = 1.0;

  if (op_info->HasAttr("enable_int8")) {
    output_scale = op_info->GetAttr<float>("output_scale");

    // X node
    std::shared_ptr<Node> x_node = nullptr;
    if (graph->Has(x_name)) {
      x_node = graph->Get(x_name);
    } else {
      LOG(FATAL) << "[NNA] Softmax: Could not find the input tensor.";
    }

    imgdnn_quant_param output_quant_param;
    output_quant_param.scale = output_scale;
    output_quant_param.zero_point = 128;
    imgdnn_tensor softmax_out_tensor = graph->GetBuilder()->createSoftmaxLayer(
        x_node->data(), 1.0, axis, output_quant_param);

    graph->Add(out_name, softmax_out_tensor, IMGDNN_TYPE_Q_U8);
  } else {
    LOG(FATAL) << "[NNA] Softmax: has no enable_int8 attribute.";
  }

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(softmax,
                         kNNA,
                         paddle::lite::subgraph::nna::SoftmaxConverter);
