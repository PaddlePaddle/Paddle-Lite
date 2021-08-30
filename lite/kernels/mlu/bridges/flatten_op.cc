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

int FlattenConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto out_var_name = op_info->Output("Out").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto output = scope->FindVar(out_var_name)->GetMutable<Tensor>();
  auto output_dims = output->dims().Vectorize();

  // ================== Trans1: NHWC => NCHW ===========================
  auto input_tensor = graph->GetNode(x_var_name);
  auto trans_1_axis = std::move(GetAxisNHWC2NCHW<int>(x->dims().size()));
  auto trans1_out = graph->AddNode(x_var_name + ".trans.i",
                                   x->dims().Vectorize(),
                                   CNML_TENSOR,
                                   CNML_NCHW,
                                   graph->FPType(),
                                   CNML_NCHW);
  cnmlBaseOp_t trans1_op{nullptr};
  cnmlNdTransposeOpParam_t trans1_param{nullptr};
  CNML_CALL(cnmlCreateNdTransposeOpParam(
      &trans1_param, trans_1_axis.data(), trans_1_axis.size()));
  CNML_CALL(cnmlCreateNdTransposeProOp(&trans1_op,
                                       input_tensor->mlu_tensor(),
                                       trans1_out->mlu_tensor(),
                                       trans1_param));
  // ======================== Trans1 End ==================================

  // ======================= Flatten op ===================================
  cnmlBaseOp_t flatten_op;
  auto trans2_input = graph->AddNode(out_var_name + ".trans.o",
                                     output_dims,
                                     CNML_TENSOR,
                                     CNML_NCHW,
                                     graph->FPType(),
                                     CNML_NCHW);
  int cnml_trans2_input_shape[4];
  CNML_CALL(
      cnmlGetTensorShape(trans2_input->mlu_tensor(), cnml_trans2_input_shape));
  cnmlReshapeOpParam_t reshape_param{nullptr};
  CNML_CALL(cnmlCreateNdReshapeOpParam(
      &reshape_param, cnml_trans2_input_shape, output->dims().size()));

  // Use cnmlCreatexxxOpForward to create op.
  CNML_CALL(cnmlCreateReshapeOp(&flatten_op,
                                reshape_param,
                                trans1_out->mlu_tensor(),
                                trans2_input->mlu_tensor()));
  // ======================= Flatten End ===================================

  // ================== Trans2: NCHW => NHWC ===============================
  auto trans_2_axis = std::move(GetAxisNCHW2NHWC<int>(output->dims().size()));
  auto output_tensor = graph->AddNode(
      out_var_name, output_dims, CNML_TENSOR, CNML_NCHW, graph->FPType());
  cnmlBaseOp_t trans2_op{nullptr};
  cnmlNdTransposeOpParam_t trans2_param{nullptr};
  CNML_CALL(cnmlCreateNdTransposeOpParam(
      &trans2_param, trans_2_axis.data(), trans_2_axis.size()));
  CNML_CALL(cnmlCreateNdTransposeProOp(&trans2_op,
                                       trans2_input->mlu_tensor(),
                                       output_tensor->mlu_tensor(),
                                       trans2_param));
  // ======================== Trans2 End ==================================

  // ============== DEBUG LOG ===============

  VLOG(6) << "x_var_name: " << x_var_name;
  VLOG(6) << "out_var_name: " << out_var_name;
  VLOG(6) << "input dim: " << x->dims();
  VLOG(6) << "output dim: " << output->dims();
  //   cnmlPrintTensor(input_tensor->mlu_tensor(), CNML_TENSOR);
  //   cnmlPrintTensor(trans1_out->mlu_tensor(), CNML_TENSOR);
  //   cnmlPrintTensor(trans2_input->mlu_tensor(), CNML_TENSOR);
  //   cnmlPrintTensor(output_tensor->mlu_tensor(), CNML_TENSOR);
  // ============== DEBUG END ===============
  graph->FuseOp(trans1_op);
  graph->FuseOp(flatten_op);
  graph->FuseOp(trans2_op);
  CNML_CALL(cnmlDestroyBaseOp(&trans1_op));
  CNML_CALL(cnmlDestroyBaseOp(&flatten_op));
  CNML_CALL(cnmlDestroyBaseOp(&trans2_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(flatten,
                         kMLU,
                         paddle::lite::subgraph::mlu::FlattenConverter);
REGISTER_SUBGRAPH_BRIDGE(flatten2,
                         kMLU,
                         paddle::lite::subgraph::mlu::FlattenConverter);
