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

int CastConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();
  auto in_dtype = op_info->GetAttr<int>("in_dtype");
  auto out_dtype = op_info->GetAttr<int>("out_dtype");

  CHECK(graph->HasNode(x_var_name));
  auto x_tensor = graph->GetNode(x_var_name);

  cnmlDataType_t out_type;
  cnmlCastType_t cast_type;
  if (in_dtype == 4 && out_dtype == 5) {
    cast_type = CNML_CAST_FLOAT16_TO_FLOAT32;
    out_type = CNML_DATA_FLOAT32;
  } else if (in_dtype == 5 && out_dtype == 4) {
    cast_type = CNML_CAST_FLOAT32_TO_FLOAT16;
    out_type = CNML_DATA_FLOAT16;
  } else {
    CHECK(0) << "Unsupported cast type";
  }

  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, out_type);

  cnmlBaseOp_t cast_op;
  CNML_CALL(cnmlCreateCastOp(&cast_op,
                             cast_type,
                             x_tensor->mlu_tensor(),
                             output_tensor->mlu_tensor()));
  graph->FuseOp(cast_op);
  CNML_CALL(cnmlDestroyBaseOp(&cast_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(cast,
                         kMLU,
                         paddle::lite::subgraph::mlu::CastConverter);
