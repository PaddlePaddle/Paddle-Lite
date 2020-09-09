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
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

int SliceConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  // input
  auto input_var_name = op_info->Input("Input").front();
  auto input = scope->FindVar(input_var_name)->GetMutable<lite::Tensor>();
  auto input_shape = input->dims().Vectorize();
  // output
  auto output_var_name = op_info->Output("Out").front();
  auto output = scope->FindVar(output_var_name)->GetMutable<lite::Tensor>();
  // attr
  auto axes = op_info->GetAttr<std::vector<int32_t>>("axes");
  auto starts = op_info->GetAttr<std::vector<int32_t>>("starts");
  auto ends = op_info->GetAttr<std::vector<int32_t>>("ends");

  CHECK(graph->HasNode(input_var_name));
  auto input_tensor = graph->GetNode(input_var_name);
  auto output_tensor = graph->AddNode(output_var_name,
                                      output->dims().Vectorize(),
                                      CNML_TENSOR,
                                      CNML_NCHW,
                                      graph->FPType());

  std::vector<int32_t> begin_index(input_shape.size(), 0);
  std::vector<int32_t> end_index(input_shape.size());
  std::vector<int32_t> strides(input_shape.size(), 1);
  auto nhwc2nchw_axis = std::move(GetAxisNHWC2NCHW<int>(input_shape.size()));
  for (size_t i = 0; i < input_shape.size(); ++i) {
    end_index[nhwc2nchw_axis[i]] = input_shape[i];
  }
  for (size_t i = 0; i < axes.size(); i++) {
    int dim_value = input_shape[axes[i]];
    int end = ends[i] < 0 ? std::max(ends[i] + dim_value, 0) : ends[i];
    begin_index[nhwc2nchw_axis[axes[i]]] =
        starts[i] < 0 ? std::max(starts[i] + dim_value, 0) : starts[i];
    end_index[nhwc2nchw_axis[axes[i]]] = std::min(end, dim_value);
  }

  cnmlNdStridedSliceOpParam_t param;
  cnmlBaseOp_t slice_op;
  CNML_CALL(cnmlCreateNdStridedSliceOpParam(&param,
                                            input_shape.size(),
                                            begin_index.data(),
                                            end_index.data(),
                                            strides.data()));
  CNML_CALL(cnmlCreateNdStridedSliceOp(&slice_op,
                                       param,
                                       input_tensor->mlu_tensor(),
                                       output_tensor->mlu_tensor()));
  CNML_CALL(cnmlDestroyNdStridedSliceOpParam(&param));

  graph->FuseOp(slice_op);
  CNML_CALL(cnmlDestroyBaseOp(&slice_op));
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(slice,
                         kMLU,
                         paddle::lite::subgraph::mlu::SliceConverter);
