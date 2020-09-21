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

#include "lite/kernels/nna/bridges/graph.h"
#include "lite/kernels/nna/bridges/registry.h"
#include "lite/kernels/nna/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace nna {

int BatchNormConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NNA] Converting " + op_type + "...";

  // Get innat and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto scale_name = op_info->Input("Scale").front();
  auto scale = scope->FindMutableTensor(scale_name);
  auto bias_name = op_info->Input("Bias").front();
  auto bias = scope->FindMutableTensor(bias_name);
  auto mean_name = op_info->Input("Mean").front();
  auto mean = scope->FindMutableTensor(mean_name);
  auto variance_name = op_info->Input("Variance").front();
  auto variance = scope->FindMutableTensor(variance_name);
  auto y_name = op_info->Output("Y").front();
  // float momentum = op_info->GetAttr<float>("momentum");
  float epsilon = op_info->GetAttr<float>("epsilon");
  // int mode = 1;  // bnScale, bnBias tensor dims are 1xCx1x1
  /*
  bool use_global_stats = !op_info->HasAttr("use_global_stats") ||
                          op_info->GetAttr<bool>("use_global_stats");
  if (!use_global_stats) {
    LOG(WARNING) << "[NNA] Only use_global_stats=true is supported by DDK";
  }
  */

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    // x_node = graph->Add(x_name, *x);
    LOG(WARNING) << "BatchNormConverter:x_node not in graph";
  }

  ConvNetBuilder& builder = graph->GetBuilder();
  auto bn_out = builder.createBatchNormLayer(x_node->data(),
                                             mean->mutable_data<float>(),
                                             variance->mutable_data<float>(),
                                             epsilon);
  bn_out = builder.createScaleLayer(
      bn_out, true, scale->mutable_data<float>(), bias->mutable_data<float>());

  // PrecisionType precision = x->precision();
  imgdnn_tensor_descriptor desc;
  imgdnn_err_code err = imgdnnGetTensorDescriptor(bn_out, &desc);
  CHECK(err == IMGDNN_SUCCESS) << "fail get tensor description(BN)";

  graph->Add(y_name, bn_out, desc.type);

  return SUCCESS;
}

}  // namespace nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(batch_norm,
                         kNNA,
                         paddle::lite::subgraph::nna::BatchNormConverter);
