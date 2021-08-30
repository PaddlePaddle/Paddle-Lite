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

#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

int NormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Get input vars and op attributes
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto x_dims = x->dims().Vectorize();

  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  int axis = op_info->GetAttr<int>("axis");
  int epsilon = op_info->GetAttr<float>("epsilon");
  if (axis < 0) {
    axis = axis + x_dims.size();
  }
  std::vector<int> nchw2nhwc = {0, 3, 1, 2};
  int nhwc_axis = nchw2nhwc[axis];

  CHECK(graph->HasNode(x_var_name));
  auto input_tensor = graph->GetNode(x_var_name);
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, graph->FPType());

  // ======== DEBUG ===============
  VLOG(6) << "x name=" << x_var_name;
  VLOG(6) << "out name=" << out_var_name;
  VLOG(6) << "x dims=" << x->dims();
  VLOG(6) << "out dims=" << output->dims();
  VLOG(6) << "axis =" << axis;
  VLOG(6) << "nwhc axis=" << nhwc_axis;
  VLOG(6) << "epsilon =" << epsilon;
  // cnmlPrintTensor(input_tensor->mlu_tensor(), CNML_TENSOR);
  // cnmlPrintTensor(output_tensor->mlu_tensor(), CNML_TENSOR);
  // ======== DEBUG END ============
  cnmlBaseOp_t norm_op{nullptr};

  cnmlNormalizeOpParam_t param;
  int mode = -1;
  switch (axis) {
    case 0:
      mode = 3;  // N
      break;
    case 1:
      mode = 0;  // C
      break;
    case 2:
      mode = 4;  // H
      break;
    case 3:
      mode = 5;  // W
      break;
    default:
      CHECK(0);
      break;
  }
  cnmlCreateNormalizeOpParamV2(&param,
                               0,  // p
                               0,  // use_scale
                               mode,
                               1,  // weight
                               epsilon);

  CNML_CALL(cnmlCreateNormalizeOp(&norm_op,
                                  param,
                                  input_tensor->mlu_tensor(),
                                  output_tensor->mlu_tensor(),
                                  nullptr,
                                  false /*is_fix8_mode*/));
  graph->FuseOp(norm_op);
  CNML_CALL(cnmlDestroyBaseOp(&norm_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(norm,
                         kMLU,
                         paddle::lite::subgraph::mlu::NormConverter);
