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

std::vector<int64_t> CvtYShape(const Tensor& x, Tensor* y, int axis) {
  auto x_dims = x.dims();
  // CHECK_EQ(x_dims.size(), 4UL) << "[MLU] Only support 4-dimension x";
  auto y_dims = y->dims();
  CHECK_GE(x_dims.size(), y_dims.size());

  if (axis < 0) {
    axis += x_dims.size();
  }

  std::vector<int64_t> y_new_shape(y_dims.Vectorize());
  if (y_new_shape.size() == 4UL) {
    return y_new_shape;
  }
  for (int i = 0; i < axis; i++) {
    y_new_shape.insert(y_new_shape.begin(), 1);
  }
  while (y_new_shape.size() < 4) {
    y_new_shape.push_back(1);
  }
  CHECK_EQ(y_new_shape.size(), 4UL);
  return y_new_shape;
}

int ElementwiseConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("X").front();
  auto y_var_name = op_info->Input("Y").front();
  auto out_var_name = op_info->Output("Out").front();
  auto axis = op_info->GetAttr<int>("axis");

  auto x_tensor = graph->GetNode(x_var_name);
  auto x = scope->FindTensor(x_var_name);
  std::shared_ptr<MLUTensor> y_tensor;
  if (graph->HasNode(y_var_name)) {
    y_tensor = graph->GetNode(y_var_name);
  } else {
    auto y = scope->FindMutableTensor(y_var_name);
    auto y_new_shape = CvtYShape(*x, y, axis);
    // all subgraph input tensor are built at first
    // If we can not find the tensor, it should be const tensor
    y_tensor = graph->AddNode(
        y_var_name, y_new_shape, CNML_CONST, CNML_NCHW, graph->FPType());
    graph->BindConstData(y_var_name, y);
  }

  auto output_tensor = graph->AddNode(out_var_name,
                                      x->dims().Vectorize(),
                                      CNML_TENSOR,
                                      CNML_NCHW,
                                      graph->FPType());

  cnmlBaseOp_t elementwise_op;
  if (op_type == "elementwise_add") {
    CNML_CALL(cnmlCreateBroadcastAddOp(&elementwise_op,
                                       x_tensor->mlu_tensor(),
                                       y_tensor->mlu_tensor(),
                                       output_tensor->mlu_tensor()));
  } else if (op_type == "fusion_elementwise_add_activation") {
    auto mid_tensor = graph->AddNode(out_var_name + "_mid",
                                     x->dims().Vectorize(),
                                     CNML_TENSOR,
                                     CNML_NCHW,
                                     graph->FPType());
    CNML_CALL(cnmlCreateBroadcastAddOp(&elementwise_op,
                                       x_tensor->mlu_tensor(),
                                       y_tensor->mlu_tensor(),
                                       mid_tensor->mlu_tensor()));
  } else if (op_type == "elementwise_sub") {
    CNML_CALL(cnmlCreateBroadcastSubOp(&elementwise_op,
                                       x_tensor->mlu_tensor(),
                                       y_tensor->mlu_tensor(),
                                       output_tensor->mlu_tensor()));
  } else if (op_type == "elementwise_mul") {
    CNML_CALL(cnmlCreateBroadcastMultOp(&elementwise_op,
                                        x_tensor->mlu_tensor(),
                                        y_tensor->mlu_tensor(),
                                        output_tensor->mlu_tensor()));
  } else if (op_type == "elementwise_div") {
    CNML_CALL(cnmlCreateRealDivOp(&elementwise_op,
                                  x_tensor->mlu_tensor(),
                                  y_tensor->mlu_tensor(),
                                  output_tensor->mlu_tensor()));
  } else {
    LOG(WARNING) << "[MLU] Unsupported op type: " << op_type;
    return FAILED;
  }

  graph->FuseOp(elementwise_op);
  CNML_CALL(cnmlDestroyBaseOp(&elementwise_op));
  cnmlBaseOp_t act_op;
  if (op_type == "fusion_elementwise_add_activation") {
    auto mid_tensor = graph->GetNode(out_var_name + "_mid");
    auto type_string = op_info->GetAttr<std::string>("act_type");
    cnmlActiveFunction_t act_type = OpTypeToCNMLActType(type_string);
    CNML_CALL(cnmlCreateActiveOp(&act_op,
                                 act_type,
                                 mid_tensor->mlu_tensor(),
                                 output_tensor->mlu_tensor()));
    graph->FuseOp(act_op);
    CNML_CALL(cnmlDestroyBaseOp(&act_op));
  }
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(elementwise_add,
                         kMLU,
                         paddle::lite::subgraph::mlu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(fusion_elementwise_add_activation,
                         kMLU,
                         paddle::lite::subgraph::mlu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_sub,
                         kMLU,
                         paddle::lite::subgraph::mlu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_mul,
                         kMLU,
                         paddle::lite::subgraph::mlu::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(elementwise_div,
                         kMLU,
                         paddle::lite::subgraph::mlu::ElementwiseConverter);
