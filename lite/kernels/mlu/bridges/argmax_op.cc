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

int ArgmaxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
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

  int axis = op_info->GetAttr<int64_t>("axis");
  if (axis < 0) {
    axis = axis + x_dims.size();
  }
  cnmlDimension_t argmax_mode = static_cast<cnmlDimension_t>(axis);
  auto mlu_output_dim = x->dims().Vectorize();
  // shape is NCHW, layout is NHWC
  mlu_output_dim[axis] = 1;
  auto input_tensor = graph->GetNode(x_var_name);
  // if use_fp16 and axis is not c, cast input datatype from fp16 to fp32, so
  // output datatype is int32
  bool cast_to_fp32 =
      graph->FPType() == CNML_DATA_FLOAT16 && argmax_mode != CNML_DIM_C;
  cnmlBaseOp_t cast_op{nullptr};
  std::shared_ptr<MLUTensor> fp32_input_tensor;
  if (cast_to_fp32) {
    fp32_input_tensor = graph->AddNode(x_var_name + ".fp32",
                                       x_dims,
                                       CNML_TENSOR,
                                       CNML_NCHW,
                                       CNML_DATA_FLOAT32);
    cnmlCreateCastOp(&cast_op,
                     CNML_CAST_FLOAT16_TO_FLOAT32,
                     input_tensor->mlu_tensor(),
                     fp32_input_tensor->mlu_tensor());
  }
  auto output_tensor = graph->AddNode(
      out_var_name, mlu_output_dim, CNML_TENSOR, CNML_NCHW, CNML_DATA_INT32);

  CHECK(graph->HasNode(x_var_name));
  cnmlBaseOp_t argmax_op{nullptr};
  // ======================= DEBUG INFO =====================
  VLOG(6) << "x_var_name: " << x_var_name;
  VLOG(6) << "out_var_name: " << out_var_name;
  VLOG(6) << "x dims: " << x->dims();
  VLOG(6) << "output dims: " << output->dims();
  VLOG(6) << "axis: " << axis;
  VLOG(6) << "cast_to_fp32: " << cast_to_fp32;
  cnmlPrintTensor(input_tensor->mlu_tensor(), CNML_TENSOR);
  cnmlPrintTensor(output_tensor->mlu_tensor(), CNML_TENSOR);
  // ======================= DEBUG END =====================

  CNML_CALL(cnmlCreateArgmaxOp(&argmax_op,
                               argmax_mode,
                               cast_to_fp32 ? fp32_input_tensor->mlu_tensor()
                                            : input_tensor->mlu_tensor(),
                               output_tensor->mlu_tensor()));
  if (cast_to_fp32) {
    graph->FuseOp(cast_op);
  }
  graph->FuseOp(argmax_op);
  CNML_CALL(cnmlDestroyBaseOp(&argmax_op));
  if (cast_op) {
    CNML_CALL(cnmlDestroyBaseOp(&cast_op));
  }
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(arg_max,
                         kMLU,
                         paddle::lite::subgraph::mlu::ArgmaxConverter);
