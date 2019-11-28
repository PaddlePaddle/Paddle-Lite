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

#include "lite/backends/npu/builder.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {
namespace bridges {

node_map_type PoolConverter(const std::shared_ptr<lite::OpLite> pool_op,
                            const node_map_type& inputs_map) {
  auto scope = pool_op->scope();
  auto op_info = pool_op->op_info();
  auto op_type = op_info->Type();
  auto unique_op_type = lite::npu::UniqueName(op_type);
  LOG(INFO) << "[NPU] Converting " + op_type + "...";

  std::shared_ptr<ge::op::Pooling> pool_node =
      std::make_shared<ge::op::Pooling>(unique_op_type);
  auto x_var_name = op_info->Input("X").front();
  auto pooling_type = op_info->GetAttr<std::string>("pooling_type");
  int npu_mode = 0;
  if (pooling_type == "max") {
    npu_mode = 0;
  } else if (pooling_type == "avg") {
    npu_mode = 1;
    CHECK(op_info->GetAttr<bool>("exclusive"))
        << "[NPU] exclusive must be true in HiAI DDK";
  } else {
    LOG(FATAL) << "[NPU] Unsupported pooling type: " << pooling_type;
  }
  bool npu_global_pooling = op_info->GetAttr<bool>("global_pooling");
  auto ksize = op_info->GetAttr<std::vector<int>>("ksize");
  auto npu_window = ge::AttrValue::LIST_INT(ksize.begin(), ksize.end());

  auto padding = op_info->GetAttr<std::vector<int>>("paddings");
  bool pads_equal = (padding[0] == padding[1]) && (padding[2] == padding[3]);
  if (!pads_equal) {
    LOG(FATAL)
        << "padding requires pad_left == pad_right, pad_top == pad_bottom";
  }
  auto npu_pad =
      ge::AttrValue::LIST_INT{padding[0], padding[1], padding[2], padding[3]};
  auto strides = op_info->GetAttr<std::vector<int>>("strides");
  auto npu_stride = ge::AttrValue::LIST_INT(strides.begin(), strides.end());
  int npu_ceil_mode = 0;
  if (op_info->HasAttr("ceil_mode")) {
    npu_ceil_mode = op_info->GetAttr<bool>("ceil_mode") ? 1 : 0;
  }

  pool_node->set_input_x(*inputs_map.at(x_var_name));
  pool_node->set_attr_mode(npu_mode);
  pool_node->set_attr_pad_mode(0);
  pool_node->set_attr_global_pooling(npu_global_pooling);
  pool_node->set_attr_window(npu_window);
  pool_node->set_attr_pad(npu_pad);
  pool_node->set_attr_stride(npu_stride);
  pool_node->set_attr_ceil_mode(npu_ceil_mode);
  // output_node->set_attr_data_mode(npu_data_mode);

  lite::npu::OpList::Global().add(inputs_map.at(x_var_name));
  lite::npu::OpList::Global().add(pool_node);

  node_map_type outputs_map;
  outputs_map[op_info->Output("Out").front()] = pool_node;
  return outputs_map;
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_BRIDGE(pool2d, paddle::lite::kernels::npu::bridges::PoolConverter);
