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

#include "lite/operators/pool_op.h"
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

inline cnmlPoolMode_t ToCnmlPoolMode(const std::string& pool_mode) {
  cnmlPoolMode_t cnml_pool_mode;
  if (pool_mode == "max") {
    cnml_pool_mode = CNML_POOL_MAX;
  } else if (pool_mode == "avg") {
    cnml_pool_mode = CNML_POOL_AVG;
  } else {
    CHECK(false) << "Unexpected pool mode " << pool_mode;
  }

  return cnml_pool_mode;
}

int PoolConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // Get input, and attributes
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindTensor(x_var_name);
  auto input_dims_nhwc = x->dims();
  const auto input_dims = DimNHWC2NCHW(input_dims_nhwc);
  auto output_var_name = op_info->Output("Out").front();
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  auto ceil_mode = op_info->GetAttr<bool>("ceil_mode");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  auto strides = op_info->GetAttr<std::vector<int>>("strides");

  if (paddings.size() == 2L) {
    for (size_t i = 0; i < 2L; ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  int pad_height = paddings[0];
  int pad_width = paddings[2];
  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }
  bool adaptive = false;
  if (op_info->HasAttr("adaptive")) {
    adaptive = op_info->GetAttr<bool>("adaptive");
  }
  lite::operators::UpdatePadding(&paddings,
                                 global_pooling,
                                 adaptive,
                                 padding_algorithm,
                                 x->dims(),
                                 strides,
                                 ksize);

  std::vector<int64_t> output_shape({input_dims[0], input_dims[1]});
  for (size_t i = 0; i < 2; i++) {
    output_shape.push_back(
        (input_dims[i + 2] + paddings[2 * i] + paddings[2 * i + 1] - ksize[0]) /
            strides[i] +
        1);
  }

  auto output_shape_nhwc = DimNCHW2NHWC(output_shape);
  auto output_tensor = graph->AddNode(output_var_name,
                                      output_shape_nhwc,
                                      CNML_TENSOR,
                                      CNML_NHWC,
                                      graph->FPType());
  scope->FindVar(output_var_name)
      ->GetMutable<::paddle::lite::Tensor>()
      ->Resize(output_shape_nhwc);

  cnmlPoolOpParam_t pool_param;
  CNML_CALL(
      cnmlCreatePoolOpParam_V2(&pool_param,
                               ksize[0],
                               ksize[1],
                               strides[0],
                               strides[1],
                               pad_height,
                               pad_width,
                               1,  // dilation
                               1,
                               ToCnmlPoolMode(pooling_type),
                               ceil_mode ? CNML_POOL_KVALID : CNML_POOL_KFULL,
                               true, /* real */
                               1 /* blend factor */));
  cnmlBaseOp_t pool_op;
  CNML_CALL(cnmlCreatePoolOp(&pool_op,
                             pool_param,
                             graph->GetNode(x_var_name)->mlu_tensor(),
                             output_tensor->mlu_tensor()));
  CNML_CALL(cnmlDestroyPoolOpParam(&pool_param));
  graph->FuseOp(pool_op);
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pool2d,
                         kMLU,
                         paddle::lite::subgraph::mlu::PoolConverter);
