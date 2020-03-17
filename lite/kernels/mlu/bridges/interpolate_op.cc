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
  auto x_dims = x->dims();
  CHECK_EQ(x_dims.size(), 4);
  auto scale = op_info->GetAttr<float>("scale");
  auto out_w = op_info->GetAttr<int>("out_w");
  auto out_h = op_info->GetAttr<int>("out_h");
  auto align_corners = op_info->GetAttr<bool>("align_corners");
  /* int align_mode = */
  /*     op_info->HasAttr("align_mode") ? op_info->GetAttr<int>("align_mode") :
   * 1; */
  /* auto interp_method = op_info->GetAttr<std::string>("interp_method"); */
  /* if (align_mode == 0 && !align_corners) { */
  /*   LOG(WARNING) << "[NPU] align_mode = 0 && " */
  /*                   "align_corners = false isn't " */
  /*                   "supported in CNML"; */
  /*   return FAILED; */
  /* } */

  CHECK(graph->HasNode(x_var_name));
  auto input_tensor = graph->GetNode(x_var_name);
  auto out = scope->FindVar(out_var_name)->GetMutable<Tensor>();

  /* int x_h, x_w; */
  /* if (interp_method == "bilinear") { */
  /*   x_h = x_dims[1]; */
  /*   x_w = x_dims[2]; */
  /* auto output_tensor = graph->AddNode( */
  /*     out_var_name, out->dims().Vectorize(), CNML_TENSOR, CNML_NHWC,
   * graph->FPType()); */
  /* } */

  auto x_h = x_dims[1];
  auto x_w = x_dims[2];
  auto output_tensor = graph->AddNode(out_var_name,
                                      out->dims().Vectorize(),
                                      CNML_TENSOR,
                                      CNML_NHWC,
                                      graph->FPType());

  // Priority: OutSize > scale > out_h/out_w
  if (scale > 0) {
    out_h = static_cast<int>(x_h * scale);
    out_w = static_cast<int>(x_w * scale);
    out_h = out_h > 0 ? out_h : -1;
    out_w = out_w > 0 ? out_w : -1;
  }

  // Update out_h and out_w and create out_size node if has OutSize
  if (HasInputArg(op_info, scope, "OutSize")) {
    auto out_size_name = op_info->Input("OutSize").front();
    auto out_size = scope->FindVar(out_size_name)->GetMutable<Tensor>();
    CHECK_EQ(out_size->numel(), 2);
    CHECK(out_size->persistable());
    auto out_size_data = out_size->mutable_data<int>();
    // Update out_h and out_w if has OutSize
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  }

  /* std::cout << "@@@scale: " << scale << "; in| w, h: " << x_w << ":" << x_h
   * << "; out| w, h: " << out_w << ":" << out_h << std::endl; */

  cnmlBaseOp_t interp_op;
  /* if (interp_method == "bilinear") { */
  /*   cnmlInterpOpParam_t interp_param; */
  /*   CNML_CALL(cnmlCreateInterpOpParam(&interp_param, out_w, out_h,
   * align_corners)); */
  /*   CNML_CALL(cnmlCreateInterpOp(&interp_op, */
  /*                                input_tensor->mlu_tensor(), */
  /*                                output_tensor->mlu_tensor(), */
  /*                                interp_param)); */
  /*   CNML_CALL(cnmlDestroyInterpOpParam(&interp_param)); */
  /* } else if (interp_method == "nearest") { */
  cnmlNearestNeighborOpParam_t nn_param;
  CNML_CALL(cnmlCreateNearestNeighborOpParam(&nn_param, out_w, out_h));
  CNML_CALL(cnmlSetNearestNeighborAlignCorner(&nn_param, align_corners));
  CNML_CALL(cnmlCreateNearestNeighborOp(&interp_op,
                                        input_tensor->mlu_tensor(),
                                        output_tensor->mlu_tensor(),
                                        nn_param));
  CNML_CALL(cnmlDestroyNearestNeighborOpParam(&nn_param));
  /* } else { */
  /*   LOG(WARNING) << "[MLU] Unsupported interpolate method: " <<
   * interp_method; */
  /*   return FAILED; */
  /* } */
  graph->FuseOp(interp_op);

  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(nearest_interp,
                         kMLU,
                         paddle::lite::subgraph::mlu::InterpolateConverter);
