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

int SoftmaxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Get op's attributes
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  auto x_shape =
      scope->FindVar(x_var_name)->GetMutable<Tensor>()->dims().Vectorize();

  // nchw axis to nhwc aixs
  std::vector<int> nchw2nhwc_axis(x_shape.size());
  nchw2nhwc_axis[0] = 0;
  if (x_shape.size() > 1) nchw2nhwc_axis[1] = x_shape.size() - 1;
  for (size_t i = 2; i < x_shape.size(); ++i) {
    nchw2nhwc_axis[i] = i - 1;
  }
  int axis = 1;
  if (op_info->HasAttr("axis")) {
    axis = op_info->GetAttr<int>("axis");
    if (axis < 0) {
      axis = output_dims.size() + axis;
    }
  }
  int nhwc_axis = nchw2nhwc_axis[axis];

  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, graph->FPType());
  cnmlBaseOp_t softmax_op;
  CNML_CALL(cnmlCreateNdSoftmaxOp(&softmax_op,
                                  nhwc_axis,
                                  graph->GetNode(x_var_name)->mlu_tensor(),
                                  output_tensor->mlu_tensor()));
  graph->FuseOp(softmax_op);
  CNML_CALL(cnmlDestroyBaseOp(&softmax_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(softmax,
                         kMLU,
                         paddle::lite::subgraph::mlu::SoftmaxConverter);
