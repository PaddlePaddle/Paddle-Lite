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

#include "lite/operators/pool_op.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int PoolConverter(void* ctx, OpLite* op) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindTensor(x_var_name);
  auto out_var_name = op_info->Output("Out").front();
  auto pool_node = graph->AddNode<ge::op::Pooling>(out_var_name);
  pool_node->set_input_x(*graph->GetNode(x_var_name));

  int mode = 0;
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  if (pooling_type == "max") {
    mode = 0;
  } else if (pooling_type == "avg") {
    mode = 1;
    CHECK(op_info->GetAttr<bool>("exclusive"))
        << "[NPU] exclusive must be true in HiAI DDK";
  } else {
    LOG(WARNING) << "[NPU] Unsupported pooling type: " << pooling_type;
    return FAILED;
  }
  pool_node->set_attr_mode(mode);

  int pad_mode = 0;
  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }
  if (padding_algorithm == "SAME") {
    pad_mode = 6;
  } else if (padding_algorithm == "VALID") {
    pad_mode = 5;
  }
  pool_node->set_attr_pad_mode(pad_mode);

  bool global_pooling = op_info->GetAttr<bool>("global_pooling");
  pool_node->set_attr_global_pooling(global_pooling);

  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  pool_node->set_attr_window(
      ge::AttrValue::LIST_INT(ksize.begin(), ksize.end()));

  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  if (paddings.size() == 2L) {
    for (size_t i = 0; i < 2L; ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
      << "[NPU] Paddings size should be the same or twice as the inputs size.";
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
  pool_node->set_attr_pad(ge::AttrValue::LIST_INT{
      paddings[0], paddings[1], paddings[2], paddings[3]});
  pool_node->set_attr_stride(
      ge::AttrValue::LIST_INT(strides.begin(), strides.end()));

  int ceil_mode = 0;
  if (op_info->HasAttr("ceil_mode")) {
    ceil_mode = op_info->GetAttr<bool>("ceil_mode") ? 1 : 0;
  }
  pool_node->set_attr_ceil_mode(ceil_mode);
  // pool_node->set_attr_data_mode(data_mode);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(NPU,
                         pool2d,
                         paddle::lite::subgraph::npu::PoolConverter);
