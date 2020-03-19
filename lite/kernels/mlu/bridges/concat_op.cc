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

#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

int ConcatConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X");
  auto out_var_name = op_info->Output("Out").front();
  auto param_axis = op_info->GetAttr<int>("axis");
  // auto x = scope->FindVar(x_var_name[0])->GetMutable<Tensor>();

  auto input_num = x_var_name.size();

  std::vector<cnmlTensor_t> input_tensor;
  std::vector<std::vector<int64_t>> input_dims;
  for (auto x_name : x_var_name) {
    CHECK(graph->HasNode(x_name));
    input_tensor.push_back(graph->GetNode(x_name)->mlu_tensor());
    auto x = scope->FindVar(x_name)->GetMutable<Tensor>();
    input_dims.push_back(x->dims().Vectorize());
  }

  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  int axis = (param_axis < 0) ? (param_axis + output->dims().size()) : param_axis;
  int nchw_to_nhwc_axis_map[4] = {0, 3, 1, 2};
  int nhwc_axis = nchw_to_nhwc_axis_map[axis];

  std::vector<int64_t> output_dims;
  output_dims.assign(output->dims().size(), 0);

  /* std::cout << string_format("concat axis: %d(NCHW), %d(NHWC)", axis, nhwc_axis) << std::endl; */
  
  for (int i = 0; i < output_dims.size(); ++i) {
    if (i == nhwc_axis) {
      for (auto &dim : input_dims) output_dims[i] += dim[i];
    } else {
      output_dims[i] = input_dims[0][i];
    }
  }

  /* std::cout << string_format("concat output dim: %ld, %ld, %ld, %ld") << std::endl; */

  output->Resize(output_dims);
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NHWC, graph->FPType());

  cnmlBaseOp_t concat_op;
  cnmlTensor_t outputs[1];
  outputs[0] = output_tensor->mlu_tensor();
  CNML_CALL(cnmlCreateNdConcatOp(
      &concat_op, nhwc_axis, input_tensor.data(), input_num, outputs, 1));
  graph->FuseOp(concat_op);
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(concat,
                         kMLU,
                         paddle::lite::subgraph::mlu::ConcatConverter);
