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

#include "lite/kernels/npu/bridges/test_helper.h"
#include <utility>
#include "lite/backends/npu/builder.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {
namespace bridges {

void LauchOp(const std::shared_ptr<lite::OpLite> op,
             const std::vector<std::string>& input_var_names,
             const std::vector<std::string>& output_var_names) {
  auto scope = op->scope();
  auto op_type = op->op_info()->Type();

  // convert op to IR graph
  const auto& bridges = lite::kernels::npu::bridges::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType(op_type));

  node_map_type inputs_map;
  for (auto input_var_name : input_var_names) {
    auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
    ge::TensorDesc input_desc(
        ge::Shape(input->dims().Vectorize()), ge::FORMAT_NCHW, ge::DT_FLOAT);
    auto input_node = std::make_shared<ge::op::Data>(input_var_name);
    input_node->update_input_desc_x(input_desc);
    lite::npu::OpList::Global().add(input_node);
    inputs_map[input_var_name] = input_node;
  }
  auto outputs_map = supported_lists.at(op_type)(op, inputs_map);
  CHECK_GT(outputs_map.size(), 0);

  // compile IR graph to om model
  std::vector<ge::Operator> subgraph_inputs;
  for (auto input_var_name : input_var_names) {
    subgraph_inputs.push_back(*inputs_map[input_var_name]);
  }
  std::vector<ge::Operator> subgraph_outputs;
  for (auto output_var_name : output_var_names) {
    subgraph_outputs.push_back(*outputs_map[output_var_name]);
  }
  std::string weight_var_name = "weight";
  auto weight = scope->Var(weight_var_name)->GetMutable<Tensor>();
  weight->set_persistable(true);
  weight->set_precision(PRECISION(kInt8));
  CHECK(lite::npu::BuildModel(subgraph_inputs, subgraph_outputs, weight));
  CHECK_GT(weight->numel(), 0);
  CHECK_NE(weight->data<uint8_t>(), 0);

  // create subgraph op and set inputs and outputs
  cpp::OpDesc subgraph_op_desc;
  subgraph_op_desc.SetType("subgraph");
  subgraph_op_desc.SetInput("Inputs", input_var_names);
  subgraph_op_desc.SetInput("Weight", {weight_var_name});
  subgraph_op_desc.SetOutput("Outputs", output_var_names);

  auto subgraph_op =
      std::make_shared<operators::SubgraphOp>(subgraph_op_desc.Type());
  subgraph_op->SetValidPlaces({Place{TARGET(kNPU), PRECISION(kFloat)}});
  CHECK(subgraph_op->Attach(subgraph_op_desc, scope));
  CHECK(subgraph_op->CheckShape());
  CHECK(subgraph_op->InferShape());

  // create subgraph op kernel and set NPU context
  auto subgraph_kernels =
      subgraph_op->CreateKernels({Place{TARGET(kNPU), PRECISION(kFloat)}});
  CHECK(!subgraph_kernels.empty());
  auto subgraph_kernel =
      std::move(subgraph_kernels.front());  // use the first kernel by default
  auto subgraph_device = ContextScheduler::Global().NewContext(TARGET(kNPU));
  subgraph_kernel->SetContext(std::move(subgraph_device));

  // perform subgraph op kernel and store to output variables
  subgraph_kernel->Launch();

  // release all of resources of generated model
  lite::npu::OpList::Global().clear();
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(subgraph);
USE_LITE_KERNEL(subgraph, kNPU, kFloat, kNCHW, def);
