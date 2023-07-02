// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/__xpu__matmul_scale_softmax_v1_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XpuMatmulScaleSoftmaxV1Op::CheckShape() const {
  CHECK_OR_FALSE(param_.mat_q);
  CHECK_OR_FALSE(param_.mat_k);
  CHECK_OR_FALSE(param_.mat_v);
  CHECK_OR_FALSE(param_.output);

  const auto mat_q_dims = param_.mat_q->dims();
  const auto mat_k_dims = param_.mat_q->dims();
  const auto mat_v_dims = param_.mat_q->dims();
  if (mat_q_dims.size() == 3UL) {
    CHECK_EQ_OR_FALSE(mat_q_dims.size(), 3UL);
    CHECK_EQ_OR_FALSE(mat_k_dims.size(), 3UL);
    CHECK_EQ_OR_FALSE(mat_v_dims.size(), 3UL);
    CHECK_EQ(mat_k_dims[1], mat_v_dims[1]) << mat_k_dims[1] << ", "
                                           << mat_v_dims[1];
    CHECK_EQ(mat_q_dims[2], mat_k_dims[2]) << mat_q_dims[2] << ", "
                                           << mat_k_dims[2];
    CHECK_EQ(mat_k_dims[2], mat_v_dims[2]) << mat_k_dims[2] << ", "
                                           << mat_v_dims[2];
  } else {
    CHECK_EQ_OR_FALSE(mat_q_dims.size(), 4UL);
    CHECK_EQ_OR_FALSE(mat_k_dims.size(), 4UL);
    CHECK_EQ_OR_FALSE(mat_v_dims.size(), 4UL);
    CHECK_EQ(mat_q_dims[1], mat_k_dims[1]) << mat_q_dims[1] << ", "
                                           << mat_k_dims[1];
    CHECK_EQ(mat_k_dims[1], mat_v_dims[1]) << mat_k_dims[1] << ", "
                                           << mat_v_dims[1];
    CHECK_EQ(mat_k_dims[2], mat_v_dims[2]) << mat_k_dims[2] << ", "
                                           << mat_v_dims[2];
    CHECK_EQ(mat_q_dims[3], mat_k_dims[3]) << mat_q_dims[3] << ", "
                                           << mat_k_dims[3];
    CHECK_EQ(mat_k_dims[3], mat_v_dims[3]) << mat_k_dims[3] << ", "
                                           << mat_v_dims[3];
  }
  return true;
}

bool XpuMatmulScaleSoftmaxV1Op::InferShapeImpl() const {
  const auto& input_dims = param_.mat_q->dims();
  param_.output->Resize(input_dims);
  param_.output->set_lod(param_.mat_q->lod());
  return true;
}

bool XpuMatmulScaleSoftmaxV1Op::AttachImpl(const cpp::OpDesc& op_desc,
                                           lite::Scope* scope) {
  param_.mat_q =
      scope->FindVar(op_desc.Input("mat_q").front())->GetMutable<Tensor>();
  param_.mat_k =
      scope->FindVar(op_desc.Input("mat_k").front())->GetMutable<Tensor>();
  param_.mat_v =
      scope->FindVar(op_desc.Input("mat_v").front())->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();
  param_.alpha = op_desc.GetAttr<float>("alpha");
  param_.matmul_trans_info =
      op_desc.GetAttr<std::vector<int>>("MatmulTransInfo");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__matmul_scale_softmax_v1,
                 paddle::lite::operators::XpuMatmulScaleSoftmaxV1Op);
