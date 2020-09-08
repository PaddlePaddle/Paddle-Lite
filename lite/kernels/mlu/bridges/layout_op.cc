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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

int LayoutConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("Input").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  std::shared_ptr<MLUTensor> output_tensor;

  CHECK(graph->HasNode(x_var_name));
  std::vector<int> axis;
  auto x_tensor = graph->GetNode(x_var_name);
  auto x_data_order = x_tensor->dorder();
  auto x_dims = x->dims().Vectorize();
  if (x_data_order == CNML_NCHW) {
    switch (x_dims.size()) {
      case 2:
        axis = {0, 1};
        break;
      case 3:
        axis = {0, 2, 1};
        break;
      case 4:
        axis = {0, 2, 3, 1};
        break;
      case 5:
        axis = {0, 2, 3, 4, 1};
        break;
      default:
        CHECK(0) << "Unsupport shape";
    }
    output_tensor = graph->AddNode(
        out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, x_tensor->dtype());
    VLOG(3) << "layout transpose nchw to nhwc" << std::endl;
  } else {
    switch (x_dims.size()) {
      case 2:
        axis = {0, 1};
        break;
      case 3:
        axis = {0, 2, 1};
        break;
      case 4:
        axis = {0, 3, 1, 2};
        break;
      case 5:
        axis = {0, 4, 1, 2, 3};
        break;
      default:
        CHECK(0) << "Unsupport shpae";
    }
    VLOG(3) << "layout transpose nhwc to nchw" << std::endl;
    output_tensor = graph->AddNode(out_var_name,
                                   output_dims,
                                   CNML_TENSOR,
                                   CNML_NCHW,
                                   x_tensor->dtype(),
                                   CNML_NCHW);
  }
  cnmlBaseOp_t layout_op;
  cnmlNdTransposeOpParam_t transpose_param;
  CNML_CALL(
      cnmlCreateNdTransposeOpParam(&transpose_param, axis.data(), axis.size()));
  CNML_CALL(cnmlCreateNdTransposeProOp(&layout_op,
                                       x_tensor->mlu_tensor(),
                                       output_tensor->mlu_tensor(),
                                       transpose_param));
  graph->FuseOp(layout_op);
  CNML_CALL(cnmlDestroyBaseOp(&layout_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(layout,
                         kMLU,
                         paddle::lite::subgraph::mlu::LayoutConverter);
