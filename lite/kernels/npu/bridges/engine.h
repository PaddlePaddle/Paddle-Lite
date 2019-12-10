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

#pragma once

#include <memory>
#include <string>
#include <vector>
#include "ai_ddk_lib/include/HiAiModelManagerService.h"
#include "ai_ddk_lib/include/hiai_ir_build.h"
#include "lite/core/mir/subgraph/subgraph_engine_base.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

class Engine : public subgraph::Engine {
 public:
  Engine(int block_idx,
         cpp::BlockDesc *block_desc,
         const std::vector<std::string> &input_names,
         const std::vector<std::string> &output_names,
         Scope *scope)
      : subgraph::Engine(
            block_idx, block_desc, input_names, output_names, scope) {}

 protected:
  int BuildDeviceProgram() override;
  int LaunchDeviceProgram() override;

  std::string model_name_;
  hiai::AiContext model_context_;
  std::vector<int64_t> device_idatasizes_;
  std::vector<int64_t> device_odatasizes_;
  std::vector<std::shared_ptr<hiai::AiTensor>> device_itensors_;
  std::vector<std::shared_ptr<hiai::AiTensor>> device_otensors_;
  std::unique_ptr<hiai::AiModelMngerClient> device_program_{nullptr};
};

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
