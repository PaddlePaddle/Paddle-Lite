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

#include "lite/backends/npu/bridge/test_helper.h"
#include <utility>
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/utils.h"
#include "lite/core/op_registry.h"
#include "lite/operators/graph_op.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

void LauchOp(const std::shared_ptr<lite::OpLite> op,
             const std::vector<std::string>& input_var_names,
             const std::vector<std::string>& output_var_names) {
  auto scope = op->scope();
  auto op_type = op->op_info()->Type();

  // convert op to IR graph
  const auto& bridges = lite::npu::bridge::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType(op_type));

  node_map_type inputs_map;
  for (auto input_var_name : input_var_names) {
    auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
    ge::TensorDesc input_desc(
        ge::Shape(input->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
    auto input_node = std::make_shared<ge::op::Data>(input_var_name);
    input_node->update_input_desc_x(input_desc);
    npu::OpList::Global().add(input_node);
    inputs_map[input_var_name] = input_node;
  }
  auto outputs_map = supported_lists.at(op_type)(op, inputs_map);
  CHECK_GT(outputs_map.size(), 0);

  // compile IR graph to om model
  std::vector<ge::Operator> graph_inputs;
  for (auto input_var_name : input_var_names) {
    graph_inputs.push_back(*inputs_map[input_var_name]);
  }
  std::vector<ge::Operator> graph_outputs;
  for (auto output_var_name : output_var_names) {
    graph_outputs.push_back(*outputs_map[output_var_name]);
  }
  std::string model_name(UniqueName("test_" + op_type) + ".om");
  CHECK(npu::BuildNPUClient(graph_inputs, graph_outputs, model_name));

  // create graph op and set inputs and outputs
  cpp::OpDesc graph_op_desc;
  graph_op_desc.SetType("graph_op");
  graph_op_desc.SetInput("Inputs", input_var_names);
  graph_op_desc.SetOutput("Outputs", output_var_names);
  graph_op_desc.SetAttr("model_name", model_name);

  auto graph_op =
      std::make_shared<operators::GraphOpLite>(graph_op_desc.Type());
  graph_op->SetValidPlaces({Place{TARGET(kNPU), PRECISION(kFloat)}});
  CHECK(graph_op->Attach(graph_op_desc, scope));
  CHECK(graph_op->CheckShape());
  CHECK(graph_op->InferShape());

  // create graph op kernel and set NPU context
  auto graph_kernels =
      graph_op->CreateKernels({Place{TARGET(kNPU), PRECISION(kFloat)}});
  CHECK(!graph_kernels.empty());
  auto graph_kernel =
      std::move(graph_kernels.front());  // use the first kernel by default
  auto graph_ctx = ContextScheduler::Global().NewContext(TARGET(kNPU));
  graph_kernel->SetContext(std::move(graph_ctx));

  // perform graph op kernel and store to output variables
  graph_kernel->Launch();

  // release all of resources of generated model
  npu::OpList::Global().clear();
  npu::DeviceInfo::Global().Clear();
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(graph_op);
USE_LITE_KERNEL(graph_op, kNPU, kFloat, kNCHW, def);
