// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/huawei_ascend_npu/bridges/graph.h"
#include "lite/kernels/huawei_ascend_npu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace huawei_ascend_npu {

int CastConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[HUAWEI_ASCEND_NPU] Converting " + op_type + "...";

  // Get input, output and op attributes
  auto x_name = op_info->Input("X").front();
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();

  auto out_name = op_info->Output("Out").front();

  // BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
  // SIZE_T = 19;UINT8 = 20;INT8 = 21;
  // auto in_dtype = op_info->GetAttr<int>("in_dtype");
  auto out_dtype = op_info->GetAttr<int>("out_dtype");

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    x_node = graph->Add(x_name, *x);
  }

  PrecisionType ptype = PRECISION(kFloat);
  ge::DataType otype = ge::DT_FLOAT;
  switch (out_dtype) {
    case 0:  // BOOL = 0;
      ptype = PRECISION(kBool);
      otype = ge::DT_BOOL;
      break;
    case 1:  // INT16 = 1
      ptype = PRECISION(kInt16);
      otype = ge::DT_INT16;
      break;
    case 2:  // INT32 = 2
      ptype = PRECISION(kInt32);
      otype = ge::DT_INT32;
      break;
    case 3:  // INT64 = 3
      ptype = PRECISION(kInt64);
      otype = ge::DT_INT64;
      break;
    case 4:  // FP16 = 4
      ptype = PRECISION(kFP16);
      otype = ge::DT_FLOAT16;
      break;
    case 5:  // FP32 = 5
      ptype = PRECISION(kFloat);
      otype = ge::DT_FLOAT;
      break;
    case 21:  // INT8 = 21
      ptype = PRECISION(kInt8);
      otype = ge::DT_INT8;
      break;
    default:
      LOG(FATAL) << "unsupported data type: " << out_dtype;
      break;
  }

  // Cast node
  auto cast_node = graph->Add<ge::op::Cast>(out_name, ptype);
  auto cast_op = cast_node->data<ge::op::Cast>();
  cast_op->set_input_x(*x_node->data());
  cast_op->set_attr_dst_type(otype);
  INPUT_UPDATE(cast_op, x, x_node);
  OUTPUT_UPDATE(cast_op, y, cast_node);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace huawei_ascend_npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    cast,
    kHuaweiAscendNPU,
    paddle::lite::subgraph::huawei_ascend_npu::CastConverter);
