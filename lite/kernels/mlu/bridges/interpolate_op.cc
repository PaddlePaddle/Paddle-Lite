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

int InterpolateConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto out = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto x_dims = x->dims();
  CHECK_EQ(x_dims.size(), 4u);
  auto scale = op_info->GetAttr<float>("scale");
  auto out_w = op_info->GetAttr<int>("out_w");
  auto out_h = op_info->GetAttr<int>("out_h");
  auto align_corners = op_info->GetAttr<bool>("align_corners");

  CHECK(graph->HasNode(x_var_name));
  auto input_tensor = graph->GetNode(x_var_name);

  auto in_h = x_dims[2];
  auto in_w = x_dims[3];

  // Priority: SizeTensor > OutSize > Scale > scale > out_h/out_w
  if (HasInputArg(op_info, scope, "SizeTensor")) {
    LOG(ERROR) << "Not support SizeTensor input now";
    CHECK(0);
  } else {
    if (HasInputArg(op_info, scope, "Scale")) {
      LOG(ERROR) << "Not support Scale input now";
      CHECK(0);
    }
    if (scale > 0) {
      out_h = static_cast<int>(in_h * scale);
      out_w = static_cast<int>(in_w * scale);
      out_h = out_h > 0 ? out_h : -1;
      out_w = out_w > 0 ? out_w : -1;
    }
    if (HasInputArg(op_info, scope, "OutSize")) {
      LOG(ERROR) << "Not support OutSize input now";
      CHECK(0);
    }
  }

  auto output_tensor = graph->AddNode(out_var_name,
                                      out->dims().Vectorize(),
                                      CNML_TENSOR,
                                      CNML_NCHW,
                                      graph->FPType());

  cnmlBaseOp_t interp_op;
  cnmlNearestNeighborOpParam_t nn_param;
  CNML_CALL(cnmlCreateNearestNeighborOpParam(&nn_param, out_w, out_h));
  CNML_CALL(cnmlSetNearestNeighborAlignCorner(&nn_param, align_corners));
  CNML_CALL(cnmlCreateNearestNeighborOp(&interp_op,
                                        input_tensor->mlu_tensor(),
                                        output_tensor->mlu_tensor(),
                                        nn_param));
  CNML_CALL(cnmlDestroyNearestNeighborOpParam(&nn_param));
  graph->FuseOp(interp_op);
  CNML_CALL(cnmlDestroyBaseOp(&interp_op));

  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(nearest_interp,
                         kMLU,
                         paddle::lite::subgraph::mlu::InterpolateConverter);
