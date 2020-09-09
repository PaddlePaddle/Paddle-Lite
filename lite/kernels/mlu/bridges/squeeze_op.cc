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

int SqueezeConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Create act node and set params from op
  auto fp_type = graph->FPType();
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, fp_type);
  CHECK(graph->HasNode(x_var_name));
  auto input_tensor = graph->GetNode(x_var_name);

  auto output_dims_nhwc = DimNCHW2NHWC(output_dims);
  std::vector<int> o_dims(output_dims.size());
  std::transform(output_dims_nhwc.cbegin(),
                 output_dims_nhwc.cend(),
                 o_dims.begin(),
                 [](DDim::value_type d) { return static_cast<int>(d); });

  cnmlReshapeOpParam_t param;
  cnmlBaseOp_t squeeze_op;
  CNML_CALL(cnmlCreateNdReshapeOpParam(&param, o_dims.data(), o_dims.size()));
  CNML_CALL(cnmlCreateReshapeOp(&squeeze_op,
                                param,
                                input_tensor->mlu_tensor(),
                                output_tensor->mlu_tensor()));
  CNML_CALL(cnmlDestroyReshapeOpParam(&param));
  graph->FuseOp(squeeze_op);
  CNML_CALL(cnmlDestroyBaseOp(&squeeze_op));

  if (op_type == "squeeze2") {
    auto xshape_var_name = op_info->Output("XShape").front();
    auto xshape = scope->FindVar(xshape_var_name)->GetMutable<Tensor>();
    auto dims_64 = xshape->dims().Vectorize();
    auto dims_64_nhwc = DimNCHW2NHWC(dims_64);
    auto xshape_tensor = graph->AddNode(
        xshape_var_name, dims_64, CNML_TENSOR, CNML_NCHW, fp_type);

    std::vector<int> xshape_dims(dims_64.size());
    std::transform(dims_64_nhwc.cbegin(),
                   dims_64_nhwc.cend(),
                   xshape_dims.begin(),
                   [](DDim::value_type d) { return static_cast<int>(d); });

    cnmlBaseOp_t squeeze2_op;
    CNML_CALL(cnmlCreateNdReshapeOpParam(
        &param, xshape_dims.data(), xshape_dims.size()));
    CNML_CALL(cnmlCreateReshapeOp(&squeeze2_op,
                                  param,
                                  input_tensor->mlu_tensor(),
                                  xshape_tensor->mlu_tensor()));
    CNML_CALL(cnmlDestroyReshapeOpParam(&param));
    graph->FuseOp(squeeze2_op);
    CNML_CALL(cnmlDestroyBaseOp(&squeeze2_op));
  }
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(squeeze,
                         kMLU,
                         paddle::lite::subgraph::mlu::SqueezeConverter);
REGISTER_SUBGRAPH_BRIDGE(squeeze2,
                         kMLU,
                         paddle::lite::subgraph::mlu::SqueezeConverter);
