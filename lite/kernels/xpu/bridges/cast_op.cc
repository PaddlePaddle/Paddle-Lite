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

#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

bool CvtDtype(int dtype, PrecisionType* ptype) {
  switch (dtype) {
    case 21:
      *ptype = PRECISION(kInt8);
      break;
    case 1:
      *ptype = PRECISION(kInt16);
      break;
    case 2:
      *ptype = PRECISION(kInt32);
      break;
    case 3:
      *ptype = PRECISION(kInt64);
      break;
    case 5:
      *ptype = PRECISION(kFloat);
      break;
    default:
      LOG(WARNING) << "[XPU] unsupported date type: " << dtype;
      return false;
  }
  return true;
}

int CastConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[XPU] Converting " + op_type + "...";

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto out_name = op_info->Output("Out").front();

  // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
  // SIZE_T = 19;UINT8 = 20;INT8 = 21;
  int in_dtype = op_info->GetAttr<int>("in_dtype");
  PrecisionType in_ptype;
  if (!CvtDtype(in_dtype, &in_ptype)) {
    return FAILED;
  }

  int out_dtype = op_info->GetAttr<int>("out_dtype");
  PrecisionType out_ptype;
  if (!CvtDtype(out_dtype, &out_ptype)) {
    return FAILED;
  }

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    CHECK(x->precision() == in_ptype)
        << "The data type of input tensor X should be "
        << PrecisionToStr(in_ptype) << ", but received "
        << PrecisionToStr(x->precision());
    x_node = graph->Add(x_name, *x);
  }

  // Cast node
  graph->Add(
      out_name,
      graph->builder_.CreateCast(*x_node->data(), CvtPrecisionType(out_ptype)),
      PrecisionType(out_ptype));

  return SUCCESS;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(cast,
                         kXPU,
                         paddle::lite::subgraph::xpu::CastConverter);
