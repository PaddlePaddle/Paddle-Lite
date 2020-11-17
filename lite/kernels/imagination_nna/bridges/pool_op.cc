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
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/imagination_nna/bridges/graph.h"
#include "lite/kernels/imagination_nna/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace imagination_nna {

int PoolConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[NNA] Converting " + op_type + "...";

  CHECK(op_info->HasAttr("enable_int8"));
  CHECK(op_info->GetAttr<bool>("enable_int8"));

  // Get input and output vars and op attributes
  auto x_name = op_info->Input("X").front();
  auto x_scale_name = "X0_scale";
  auto x = scope->FindMutableTensor(x_name);
  auto x_dims = x->dims();
  auto out_name = op_info->Output("Out").front();
  auto out_scale_name = "Out0_scale";

  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  auto global_pooling = op_info->GetAttr<bool>("global_pooling");
  std::vector<int> ksize = op_info->GetAttr<std::vector<int>>("ksize");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");

  bool exclusive = true;
  if (op_info->HasAttr("exclusive"))
    exclusive = op_info->GetAttr<bool>("exclusive");

  // for quantization
  CHECK(op_info->HasOutputScale(out_scale_name, true));
  float output_scale = op_info->GetOutputScale(out_scale_name, true)[0];

  // X node
  std::shared_ptr<Node> x_node = nullptr;
  if (graph->Has(x_name)) {
    x_node = graph->Get(x_name);
  } else {
    LOG(FATAL) << "[NNA] Pooling input not found: " << x_name;
  }

  // pool mode
  imgdnn_pooling_type img_pool_type;
  if (pooling_type == "max") {
    img_pool_type = IMGDNN_POOLING_MAX;
  } else if (pooling_type == "avg") {
    img_pool_type = IMGDNN_POOLING_AVERAGE;
  } else {
    LOG(WARNING) << "[NNA] Unsupported pooling type: " << pooling_type;
    return FAILED;
  }

  // pad mode
  std::string padding_algorithm("");
  if (op_info->HasAttr("padding_algorithm")) {
    padding_algorithm = op_info->GetAttr<std::string>("padding_algorithm");
  }
  // paddings and strides
  if (paddings.size() == 2L) {
    for (size_t i = 0; i < 2L; ++i) {
      int copy_pad = *(paddings.begin() + 2 * i);
      paddings.insert(paddings.begin() + 2 * i + 1, copy_pad);
    }
  }
  CHECK_EQ(paddings.size(), 4L)
      << "[NNA] Paddings size should be the same or twice as the inputs size.";
  bool adaptive = false;
  if (op_info->HasAttr("adaptive")) {
    adaptive = op_info->GetAttr<bool>("adaptive");
  }
  std::vector<int> strides = op_info->GetAttr<std::vector<int>>("strides");
  lite::operators::UpdatePadding(&paddings,
                                 global_pooling,
                                 adaptive,
                                 padding_algorithm,
                                 x->dims(),
                                 strides,
                                 ksize);

  unsigned int img_ksize[2] = {static_cast<unsigned int>(ksize[0]),
                               static_cast<unsigned int>(ksize[1])};

  unsigned int img_stride[2] = {static_cast<unsigned int>(strides[0]),
                                static_cast<unsigned int>(strides[1])};

  // top,left
  unsigned int pad_begin[2] = {static_cast<unsigned int>(paddings[0]),
                               static_cast<unsigned int>(paddings[2])};
  // bottom,right
  unsigned int pad_end[2] = {static_cast<unsigned int>(paddings[1]),
                             static_cast<unsigned int>(paddings[3])};

  bool ceil_mode = false;
  if (op_info->HasAttr("ceil_mode"))
    ceil_mode = op_info->GetAttr<bool>("ceil_mode");

  unsigned pool_h, pool_w;

  if (!ceil_mode) {
    pool_h = ((x_dims[2] + pad_begin[0] + pad_end[0] - img_ksize[0]) /
              img_stride[0]) +
             1;
    pool_w = ((x_dims[3] + pad_begin[1] + pad_end[1] - img_ksize[1]) /
              img_stride[1]) +
             1;
  } else {
    pool_h = ((x_dims[2] + pad_begin[0] + pad_end[0] - img_ksize[0] +
               img_stride[0] - 1) /
              img_stride[0]) +
             1;
    pool_w = ((x_dims[3] + pad_begin[1] + pad_end[1] - img_ksize[1] +
               img_stride[1] - 1) /
              img_stride[1]) +
             1;
  }

  if ((pool_h - 1) * img_stride[0] + img_ksize[0] >= x_dims[2] + pad_begin[0])
    pad_end[0] =
        (pool_h - 1) * img_stride[0] + img_ksize[0] - x_dims[2] - pad_begin[0];
  if ((pool_w - 1) * img_stride[1] + img_ksize[1] >= x_dims[3] + pad_begin[1])
    pad_end[1] =
        (pool_w - 1) * img_stride[1] + img_ksize[1] - x_dims[3] - pad_begin[1];

  if (global_pooling) {
    img_ksize[0] = x_dims[2];
    img_ksize[1] = x_dims[3];
  }

  imgdnn_quant_param output_quant_param;
  output_quant_param.scale = output_scale;
  output_quant_param.zero_point = 128;
  imgdnn_tensor pooling_out =
      graph->GetBuilder()->CreatePoolingLayer(x_node->data(),
                                              output_quant_param,
                                              img_ksize,
                                              img_stride,
                                              pad_begin,
                                              pad_end,
                                              img_pool_type,
                                              !exclusive);

  imgdnn_tensor_descriptor desc =
      graph->GetBuilder()->GetTensorDescriptor(pooling_out);

  graph->Add(out_name, pooling_out, desc.type);

  return REBUILD_WHEN_SHAPE_CHANGED;
}

}  // namespace imagination_nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    pool2d,
    kImaginationNNA,
    paddle::lite::subgraph::imagination_nna::PoolConverter);
