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

int PoolConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_type = kernel->GetInputDeclType("X");
  CHECK(x_type->precision() == PRECISION(kFloat));
  CHECK(x_type->layout() == DATALAYOUT(kNCHW));
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_type = kernel->GetOutputDeclType("Out");
  CHECK(out_type->precision() == PRECISION(kFloat));
  CHECK(out_type->layout() == DATALAYOUT(kNCHW));
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");

  // X node
  std::shared_ptr<ge::Operator> x_node = nullptr;
  if (graph->HasNode(x_name)) {
    x_node = graph->GetNode(x_name);
  } else {
    x_node = graph->AddNode(x_name, x_dims);
  }

  // pool mode
  int mode = 0;
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

  // pad mode
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

  // paddings and strides
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

  // ceil mode
  int ceil_mode = 0;
  if (op_info->HasAttr("ceil_mode")) {
    ceil_mode = op_info->GetAttr<bool>("ceil_mode") ? 1 : 0;
  }

  // Pooling node
  auto pool_node = graph->AddNode<ge::op::Pooling>(out_name);
  pool_node->set_input_x(*x_node);
  pool_node->set_attr_mode(mode);
  pool_node->set_attr_pad_mode(pad_mode);
  pool_node->set_attr_global_pooling(global_pooling);
  pool_node->set_attr_window(
      ge::AttrValue::LIST_INT(ksize.begin(), ksize.end()));
  pool_node->set_attr_pad(ge::AttrValue::LIST_INT{
      paddings[0], paddings[1], paddings[2], paddings[3]});
  pool_node->set_attr_stride(
      ge::AttrValue::LIST_INT(strides.begin(), strides.end()));
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
