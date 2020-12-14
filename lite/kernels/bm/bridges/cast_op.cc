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

#include <bmcompiler_defs.h>
#include <bmcompiler_if.h>
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/bm/bridges/graph.h"
#include "lite/kernels/bm/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace bm {

bool CvtDtype(int dtype, int* ptype) {
  switch (dtype) {
    case 21:
      *ptype = DTYPE_INT8;
      break;
    case 1:
      *ptype = DTYPE_INT16;
      break;
    case 2:
    case 3:
      *ptype = DTYPE_INT32;
      break;
    case 5:
      *ptype = DTYPE_FP32;
      break;
    default:
      LOG(WARNING) << "[BM] unsupported date type: " << dtype;
      return false;
  }
  return true;
}

int CastConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto x_var_name = op_info->Input("X").front();
  auto x = scope->FindVar(x_var_name)->GetMutable<lite::Tensor>();
  auto x_dims = x->dims();
  auto output_var_name = op_info->Output("Out").front();
  std::vector<int32_t> i_x_shape_data(x_dims.size());
  for (size_t i = 0; i < x_dims.size(); i++) {
    i_x_shape_data[i] = static_cast<int>(x_dims[i]);
  }

  int in_dtype = op_info->GetAttr<int>("in_dtype");
  int out_dtype = op_info->GetAttr<int>("out_dtype");
  if (in_dtype == out_dtype) {
    add_identity_layer(graph->GetCompilerHandle(),
                       static_cast<const char*>(x_var_name.c_str()),
                       const_cast<const int*>(&i_x_shape_data[0]),
                       x_dims.size(),
                       static_cast<const char*>(output_var_name.c_str()));
  } else {
    int out_bm_dtype = 0;
    int in_bm_dtype = 0;
    CHECK_EQ(CvtDtype(out_dtype, &out_bm_dtype), true);
    CHECK_EQ(CvtDtype(in_dtype, &in_bm_dtype), true);
    if (out_bm_dtype == DTYPE_FP32) {
      in_bm_dtype = out_bm_dtype;
    }
    add_dtype_convert_layer(graph->GetCompilerHandle(),
                            const_cast<const int*>(&i_x_shape_data[0]),
                            x_dims.size(),
                            static_cast<const char*>(x_var_name.c_str()),
                            static_cast<const char*>(output_var_name.c_str()),
                            in_bm_dtype,
                            out_bm_dtype);
  }

  graph->AddNode(output_var_name);
  return SUCCESS;
}

}  // namespace bm
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(cast, kBM, paddle::lite::subgraph::bm::CastConverter);
