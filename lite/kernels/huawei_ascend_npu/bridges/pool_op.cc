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

#include "lite/operators/pool_op.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int PoolConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");

  CHECK_EQ(op_info->GetAttr<bool>("exclusive"), true)
      << "[HUAWEI_ASCEND_NPU] Only exclusive=true is supported for Huawei "
         "Ascend NPU DDK.";

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  // pool mode: 0:max pooling or 1:avg pooling
  int mode = 0;
  if (pooling_type == "max") {
    mode = 0;
  } else if (pooling_type == "avg") {
    mode = 1;
  } else {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Unsupported pooling type: "
                 << pooling_type;
    return FAILED;
  }

  // pad algorithm
  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }

  // paddings and strides
  if (paddings.size() == 2L) {
    for (size_t i = 0; i < 2L; ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L) << "[HUAWEI_ASCEND_NPU] Paddings size should "
                                   "be the same or twice as the inputs size.";
  bool adaptive = false;
  if (op_info->HasAttr("adaptive")) {
    adaptive = op_info->GetAttr<bool>("adaptive");
  }
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  lite::operators::UpdatePadding(&paddings,
                                 global_pooling,
                                 adaptive,
                                 padding_algorithm,
                                 x->dims(),
                                 strides,
                                 ksize);
  // Ascend restriction: padT should equals padB, and padL should equals padR
  if (paddings[0] != paddings[1]) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Padding top should equals to padding "
                    "bottom in Huawei Ascend NPU DDK, padding top is: "
                 << paddings[0] << ", padding bottom is: " << paddings[1];
    return FAILED;
  }
  if (paddings[2] != paddings[3]) {
    LOG(WARNING) << "[HUAWEI_ASCEND_NPU] Padding left should equals to padding "
                    "right in Huawei Ascend NPU DDK, padding left is: "
                 << paddings[2] << ", padding right is: " << paddings[3];
    return FAILED;
  }

  // ceil mode
  bool ceil_mode =
      op_info->HasAttr("ceil_mode") && op_info->GetAttr<bool>("ceil_mode");

  // Pooling node
  auto pool_node = graph->Add<ge::op::Pooling>(out_name);
  auto pool_op = pool_node->data<ge::op::Pooling>();
  pool_op->set_input_x(*x_node->data());
  pool_op->set_attr_mode(mode);
  pool_op->set_attr_global_pooling(global_pooling);
  pool_op->set_attr_window(ge::Operator::OpListInt({ksize[0], ksize[1]}));
  pool_op->set_attr_stride(ge::Operator::OpListInt({strides[0], strides[1]}));
  pool_op->set_attr_pad(ge::Operator::OpListInt(
      {paddings[0], paddings[1], paddings[2], paddings[3]}));

  // "0" (ceil mode) or "1" (floor mode). Defaults to "0"
  if (!ceil_mode) {
    pool_op->set_attr_ceil_mode(1);
  }
  INPUT_UPDATE(pool_op, x, x_node);
  OUTPUT_UPDATE(pool_op, y, pool_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    pool2d,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::PoolConverter);
