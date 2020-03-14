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

int FCConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto x_var_name = op_info->Input("Input").front();
  auto w_var_name = op_info->Input("W").front();
  auto output_var_name = op_info->Output("Out").front();

  // int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto w = scope->FindVar(w_var_name)->GetMutable<Tensor>();
  auto x_dims = x->dims();
  auto w_dims = w->dims();

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(w_dims.size(), 2UL);

  // Create w node
  std::vector<int64_t> w_shape{w_dims[1], w_dims[0]};
  auto w_tensor = graph->AddNode(
      w_var_name, w_shape, CNML_FILTER, CNML_NCHW, graph->FPType());

  auto input_scale = op_info->GetAttr<float>("input_scale");

  std::vector<int64_t> output_shape_nhwc({x_dims[0], 1, 1, w_dims[1]});
  auto output_tensor = graph->AddNode(output_var_name,
                                      output_shape_nhwc,
                                      CNML_TENSOR,
                                      CNML_NHWC,
                                      graph->FPType());
  scope->FindVar(output_var_name)
      ->GetMutable<::paddle::lite::Tensor>()
      ->Resize(output_shape_nhwc);

  std::string bias_var_name;
  std::shared_ptr<MLUTensor> bias_tensor;
  // Add bias node if bias tensor exists
  if (HasInputArg(op_info, scope, "Bias")) {
    bias_var_name = op_info->Input("Bias").front();
    auto bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
    auto bias_dims = bias->dims();
    CHECK(!graph->HasNode(bias_var_name));
    // CHECK_EQ(bias_dims.production(), n);

    bias_tensor = graph->AddNode(bias_var_name,
                                 bias_dims.Vectorize(),
                                 CNML_CONST,
                                 CNML_CNHW,
                                 graph->FPType());
    graph->BindConstData(bias_var_name, bias);
  }
  cnmlBaseOp_t fc_op;
  CNML_CALL(cnmlCreateMlpOp(&fc_op,
                            graph->GetNode(x_var_name)->mlu_tensor(),
                            output_tensor->mlu_tensor(),
                            w_tensor->mlu_tensor(),
                            bias_tensor ? bias_tensor->mlu_tensor() : nullptr));
  graph->SetComputingDataType(
      fc_op, graph->GetNode(x_var_name)->mlu_tensor(), 1 / input_scale);
  auto weight_scale = op_info->GetAttr<std::vector<float>>("weight_scale");

  // LOG(INFO) << "W precision " << int(w->precision());
  if (w->precision() == PrecisionType::kUnk ||
      w->precision() == PrecisionType::kInt8) {
    std::vector<float> w_dequant(w->data_size());
    dequant(w_dequant.data(),
            w->mutable_data<int8_t>(),
            1,
            w_dims[1],
            w_dims[0],
            weight_scale);
    for (int i = 0; i < w_dims[1]; i++) {
      for (int j = 0; j < w_dims[0]; j++) {
        w->mutable_data<float>()[i * w_dims[0] + j] =
            w_dequant[i + j * w_dims[1]];
      }
    }
    w->set_precision(PrecisionType::kFloat);
  } else if (w->precision() != PrecisionType::kFloat) {
    LOG(FATAL) << "UnSupported weight precision!";
  }
  // graph->BindConstData(w_var_name, w_dequant.data());
  graph->BindConstData(w_var_name, w);

  graph->SetComputingDataType(
      fc_op,
      w_tensor->mlu_tensor(),
      1 / *min_element(weight_scale.begin(), weight_scale.end()));

  graph->FuseOp(fc_op);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(fc, kMLU, paddle::lite::subgraph::mlu::FCConverter);
