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

  CHECK(!op_info->HasAttr("activation_type"));
  auto x = scope->FindVar(x_var_name)->GetMutable<Tensor>();
  auto w = scope->FindVar(w_var_name)->GetMutable<Tensor>();
  auto output = scope->FindVar(output_var_name)->GetMutable<Tensor>();
  auto x_dims = x->dims();
  auto w_dims = w->dims();

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(w_dims.size(), 2UL);

  // Create w node
  std::vector<int64_t> cnml_w_shape;
  if (x_dims.size() == 4) {
    if (x_dims[1] * x_dims[2] * x_dims[3] == w_dims[0]) {
      cnml_w_shape = {
          static_cast<int>(w_dims[1]),
          static_cast<int>(x_dims[1]),  // input_c
          static_cast<int>(x_dims[2]),  //  input_h
          static_cast<int>(x_dims[3]),  //  input_w
      };
    } else {
      LOG(FATAL)
          << "in fc op, we expect input_h * input_w * input_c == filter_c"
          << " but we got input_c = " << x_dims[1] << " input_h = " << x_dims[2]
          << " input_w = " << x_dims[3] << " filter_c = " << w_dims[0]
          << std::endl;
    }
  } else {
    cnml_w_shape = {w_dims[1], w_dims[0]};
  }

  auto w_tensor = graph->AddNode(
      w_var_name, cnml_w_shape, CNML_FILTER, CNML_NCHW, graph->FPType());

  auto input_scale = op_info->GetInputScale(x_var_name)[0];

  auto output_tensor = graph->AddNode(output_var_name,
                                      output->dims().Vectorize(),
                                      CNML_TENSOR,
                                      CNML_NCHW,
                                      graph->FPType());

  std::string bias_var_name;
  std::shared_ptr<MLUTensor> bias_tensor;
  // Add bias node if bias tensor exists
  if (HasInputArg(op_info, scope, "Bias")) {
    bias_var_name = op_info->Input("Bias").front();
    auto bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
    auto bias_dims = bias->dims().Vectorize();
    CHECK(!graph->HasNode(bias_var_name));
    if (bias_dims.size() < 4u) {
      bias_dims.insert(bias_dims.begin(), 4 - bias_dims.size(), 1);
    }
    // CHECK_EQ(bias_dims.production(), n);

    bias_tensor = graph->AddNode(
        bias_var_name, bias_dims, CNML_CONST, CNML_NHWC, graph->FPType());
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
  auto weight_scale = op_info->GetInputScale(w_var_name);

  // LOG(INFO) << "W precision " << int(w->precision());
  if (w->precision() == PrecisionType::kUnk ||
      w->precision() == PrecisionType::kInt8) {
    std::vector<float> w_dequant(w->data_size());
    if (cnml_w_shape.size() == 2) {
      dequant(w_dequant.data(),
              w->mutable_data<int8_t>(),
              1,
              cnml_w_shape[0],
              cnml_w_shape[1],
              weight_scale);
      transpose2d(w_dequant.data(),
                  w->mutable_data<float>(),
                  {static_cast<int>(cnml_w_shape[0]),
                   static_cast<int>(cnml_w_shape[1])});
    } else if (cnml_w_shape.size() == 4) {
      dequant(w_dequant.data(),
              w->mutable_data<int8_t>(),
              1,
              cnml_w_shape[0],
              cnml_w_shape[1] * cnml_w_shape[2] * cnml_w_shape[3],
              weight_scale);

      int c_o_num = cnml_w_shape[0];
      int c_i_num = cnml_w_shape[1];
      int h_i_num = cnml_w_shape[2];
      int w_i_num = cnml_w_shape[3];

      // chw == ci * hi * wi == w_dim[0]
      // first trans [chw, co] -> [co,chw]
      std::vector<float> first_trans_output(w_dequant.size());
      int chw = c_i_num * h_i_num * w_i_num;
      transpose2d(w_dequant.data(), first_trans_output.data(), {chw, c_o_num});

      // second trans [co,ci,hi,wi] -> [co,hi,wi,ci]
      transpose(first_trans_output.data(),
                w->mutable_data<float>(),
                {c_o_num, c_i_num, h_i_num, w_i_num},
                {0, 2, 3, 1});
    } else {
      LOG(FATAL) << "expect w_shape.size == 2 or 4, but got "
                 << cnml_w_shape.size() << std::endl;
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
      1 / *max_element(weight_scale.begin(), weight_scale.end()));

  graph->FuseOp(fc_op);
  CNML_CALL(cnmlDestroyBaseOp(&fc_op));
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(fc, kMLU, paddle::lite::subgraph::mlu::FCConverter);
