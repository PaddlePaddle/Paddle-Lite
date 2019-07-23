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

#include "ai_ddk_lib/include/graph/operator_reg.h"
#include "lite/npu/bridge/registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

// TODO(xxx): input and output must be vector
std::shared_ptr<ge::Operator> FCConverter(const std::shared_ptr<OpLite> op,
                                          std::shared_ptr<ge::Operator> input) {
  auto* scope = op->scope();
  auto* op_info = op->op_info();
  auto type = op_info->Type();
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

REGISTER_NPU_CVT(fc, paddle::lite::npu::bridge::FCConverter);
