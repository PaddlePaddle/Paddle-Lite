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
#include <string>
#include <vector>
#include "ai_ddk_lib/include/graph/buffer.h"
#include "ai_ddk_lib/include/graph/graph.h"
#include "ai_ddk_lib/include/graph/model.h"
#include "ai_ddk_lib/include/graph/op/all_ops.h"
#include "ai_ddk_lib/include/graph/operator.h"
#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/npu/bridge/registry.h"
#include "lite/npu/bridge/utils.h"
#include "lite/npu/npu_helper.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

node_map_type PoolConverter(const std::shared_ptr<lite::OpLite> pool_op,
                            const node_map_type& inputs_map) {
  lite::Scope* scope = pool_op->scope();
  const lite::OpInfo* op_info = pool_op->op_info();

  std::shared_ptr<ge::op::Pooling> output_node =
      std::make_shared<ge::op::Pooling>(UniqueName("pool"));
  auto x_var_name = op_info->Input("X").front();
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  int npu_mode = 0;
  if (pooling_type == "max") {
    npu_mode = 0;
  } else if (pooling_type == "avg") {
    npu_mode = 1;
  } else {
    LOG(INFO) << "npu doesn't have this type of pooling";
  }
  int npu_pad_mode = 0;
  bool npu_global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  auto npu_window = ge::AttrValue::LIST_INT(ksize.begin(), ksize.end());

  auto padding = op_info->GetAttr<std::vector<int>>("paddings");
  auto npu_pad =
      ge::AttrValue::LIST_INT{padding[0], padding[0], padding[1], padding[1]};
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto npu_stride = ge::AttrValue::LIST_INT(strides.begin(), strides.end());
  int npu_ceil_mode = 0;
  if (op_info->HasAttr("ceil_mode")) {
    npu_ceil_mode = op_info->GetAttr<bool>("ceil_mode") ? 1 : 0;
  }

  output_node->set_input_x(*inputs_map.at(x_var_name));
  output_node->set_attr_mode(npu_mode);
  output_node->set_attr_pad_mode(npu_pad_mode);
  output_node->set_attr_global_pooling(npu_global_pooling);
  output_node->set_attr_window(npu_window);
  output_node->set_attr_pad(npu_pad);
  output_node->set_attr_stride(npu_stride);
  output_node->set_attr_ceil_mode(npu_ceil_mode);
  // output_node->set_attr_data_mode(npu_data_mode);

  OpList::Global().add(inputs_map.at(x_var_name));
  OpList::Global().add(output_node);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = output_node;
  return outputs_map;
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(pool2d, paddle::lite::npu::bridge::PoolConverter);
