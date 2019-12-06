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

#include "lite/backends/npu/bridges/registry.h"
#include "lite/backends/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridges {

int FCConverter(cvt_ctx_type* ctx, OpLite* op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto op_name = ctx->UniqueName(op_type);
  VLOG(3) << "[NPU] Converting " + op_type + "...";

  auto fc_node = ctx->AddNode<ge::op::FullConnection>(op_name);

  auto x_var_name = op_info->Input("Input").front();
  auto w_var_name = op_info->Input("W").front();
  auto out_var_name = op_info->Output("Out").front();

  int in_num_col_dims = op_info->GetAttr<int>("in_num_col_dims");
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto w = scope->FindVar(w_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  auto w_dims = w->dims();

  CHECK_GE(x_dims.size(), 2UL);
  CHECK_EQ(w_dims.size(), 2UL);

  int m = x_dims.Slice(0, in_num_col_dims).production();
  int k = x_dims.Slice(in_num_col_dims, x_dims.size()).production();
  int n = w_dims[1];
  CHECK_EQ(k * n, w_dims.production());
  VLOG(3) << "[NPU] x dims: " << x_dims << " w dims: " << w_dims << " m: " << m
          << " k: " << k << " n: " << n;

  CHECK(ctx->HasNode(x_var_name));
  CHECK(!ctx->HasNode(w_var_name));

  // Reshape x to (m, k, 1, 1)
  auto reshaped_x_node = ctx->AddNode<ge::op::Reshape>(x_var_name + "_reshape");
  reshaped_x_node->set_input_tensor(*ctx->GetNode(x_var_name));
  reshaped_x_node->set_attr_shape({m, k, 1, 1});
  reshaped_x_node->set_attr_axis(0);
  fc_node->set_input_x(*reshaped_x_node);

  // Create w const node, set its shape to (k, n, 1, 1) and fill with
  // the transposed w tensor
  auto w_const_node = ctx->AddNode<ge::op::Const>(w_var_name);
  ge::TensorDesc w_const_desc(
      ge::Shape({n, k, 1, 1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorPtr w_const_tensor = std::make_shared<ge::Tensor>();
  w_const_tensor->SetTensorDesc(w_const_desc);
  auto w_data = w->mutable_data<float>();
  std::vector<float> transposed_w_data(w_dims.production());
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      transposed_w_data[j * k + i] = w_data[i * n + j];
    }
  }
  w_const_tensor->SetData(reinterpret_cast<uint8_t*>(transposed_w_data.data()),
                          transposed_w_data.size() * sizeof(float));
  w_const_node->set_attr_value(w_const_tensor);
  fc_node->set_input_w(*w_const_node);

  // Add bias node if bias tensor exists
  if (HasInputArg(op_info, scope, "Bias")) {
    auto bias_var_name = op_info->Input("Bias").front();
    auto bias = scope->FindVar(bias_var_name)->GetMutable<lite::Tensor>();
    auto bias_dims = bias->dims();
    CHECK(!ctx->HasNode(bias_var_name));
    CHECK_EQ(bias_dims.production(), n);

    auto bias_const_node = ctx->AddNode<ge::op::Const>(bias_var_name);
    bias_const_node->set_attr_value(CvtTensor(bias, {1, n, 1, 1}));
    fc_node->set_input_b(*bias_const_node);
  }

  // Reshape output of fc_node from (m, n, 1, 1) to (m, n)
  auto reshaped_fc_node = ctx->AddNode<ge::op::Reshape>(out_var_name);
  reshaped_fc_node->set_input_tensor(*fc_node);
  reshaped_fc_node->set_attr_shape({m, n});
  reshaped_fc_node->set_attr_axis(0);
  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace bridges
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(fc, paddle::lite::npu::bridges::FCConverter);
