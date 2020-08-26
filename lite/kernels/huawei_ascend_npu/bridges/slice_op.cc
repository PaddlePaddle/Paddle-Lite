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

#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int SliceConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto input_name = op_info->Input("Input").front();
  auto input = scope->FindMutableTensor(input_name);
  auto input_dims = input->dims();
  auto input_rank = static_cast<int>(input_dims.size());
  std::vector<int64_t> input_shape = input_dims.Vectorize();

  auto out_name = op_info->Output("Out").front();

  auto axes = op_info->GetAttr<std::vector<int>>("axes");
  auto starts = op_info->GetAttr<std::vector<int>>("starts");
  auto ends = op_info->GetAttr<std::vector<int>>("ends");
  CHECK_EQ(axes.size(), starts.size());
  CHECK_EQ(axes.size(), ends.size());

  // X node
  std::shared_ptr<Node> input_node = nullptr;
  if (graph->Has(input_name)) {
    input_node = graph->Get(input_name);
  } else {
    input_node = graph->Add(input_name, *input);
  }

  // Get begin/offset based on axes and starts
  std::vector<int> offset_vec(input_rank, 0);
  std::vector<int> size_vec(input_shape.begin(), input_shape.end());
  // Get begin/offset based on axes and starts
  for (int i = 0; i < axes.size(); i++) {
    auto axis = axes[i];
    CHECK_LE(axis, input_rank)
        << "[HUAWEI_ASCEND_NPU] axes value should less than input rank.";
    offset_vec[axis] = starts[i];
    size_vec[axis] = ends[i] - starts[i];
  }

  // Cast node
  auto slice_node = graph->Add<ge::op::SliceD>(out_name);
  auto slice_op = slice_node->data<ge::op::SliceD>();
  slice_op->set_input_x(*input_node->data());
  slice_op->set_attr_offsets(
      ge::Operator::OpListInt(offset_vec.begin(), offset_vec.end()));
  slice_op->set_attr_size(
      ge::Operator::OpListInt(size_vec.begin(), size_vec.end()));
  INPUT_UPDATE(slice_op, x, input_node);
  OUTPUT_UPDATE(slice_op, y, slice_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    slice,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::SliceConverter);
