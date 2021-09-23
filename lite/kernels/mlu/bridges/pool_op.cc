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
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"

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
  auto output_var_name = op_info->Output("Out").front();
  auto output_shape = scope->FindTensor(output_var_name)->dims().Vectorize();
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  auto ceil_mode = op_info->GetAttr<bool>("ceil_mode");
  auto paddings = op_info->GetAttr<std::vector<int>>("paddings");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  CHECK(!(op_info->HasAttr("exclusive") &&
          op_info->GetAttr<bool>("exclusive") == false))
      << "Unsupport param exclusive is false!";

  if (paddings.size() == 2L) {
    for (size_t i = 0; i < 2L; ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }
  bool adaptive = false;
  if (op_info->HasAttr("adaptive")) {
    adaptive = op_info->GetAttr<bool>("adaptive");
  }
  auto input_dims = x->dims();

  lite::operators::UpdatePadding(&paddings,
                                 global_pooling,
                                 adaptive,
                                 padding_algorithm,
                                 x->dims(),
                                 strides,
                                 ksize);

  if (global_pooling) {
    ksize.resize(static_cast<size_t>(input_dims.size()) - 2);
    for (size_t i = 0; i < ksize.size(); ++i) {
      ksize[i] = static_cast<int>(input_dims[i + 2]);
    }
  }

  auto output_tensor = graph->AddNode(
      output_var_name, output_shape, CNML_TENSOR, CNML_NCHW, graph->FPType());

  cnmlPoolOpParam_t pool_param;
  CNML_CALL(
      cnmlCreatePoolOpParam_V3(&pool_param,
                               ksize[0],
                               ksize[1],
                               strides[0],
                               strides[1],
                               paddings[0],
                               paddings[1],
                               paddings[2],
                               paddings[3],
                               1,  // dilation h
                               1,  // dilation w
                               ToCnmlPoolMode(pooling_type),
                               ceil_mode ? CNML_POOL_KFULL : CNML_POOL_KVALID,
                               true, /* real */
                               1 /* blend factor */));
  cnmlBaseOp_t pool_op;
  CNML_CALL(cnmlCreatePoolOp(&pool_op,
                             pool_param,
                             graph->GetNode(x_var_name)->mlu_tensor(),
                             output_tensor->mlu_tensor()));
  CNML_CALL(cnmlDestroyPoolOpParam(&pool_param));
  graph->FuseOp(pool_op);
  CNML_CALL(cnmlDestroyBaseOp(&pool_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(pool2d,
                         kMLU,
                         paddle::lite::subgraph::mlu::PoolConverter);
