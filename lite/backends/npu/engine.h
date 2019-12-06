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
#include <unordered_map>
#include <vector>
#include "ai_ddk_lib/include/HiAiModelManagerService.h"
#include "ai_ddk_lib/include/hiai_ir_build.h"
#include "lite/core/op_lite.h"
#include "lite/core/program.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace npu {

class Device {
 public:
  static Device &Global() {
    static Device x;
    return x;
  }
  Device() {}

  int freq_level() { return freq_level_; }
  int framework_type() { return framework_type_; }
  int model_type() { return model_type_; }
  int device_type() { return device_type_; }

 private:
  int freq_level_{3};
  int framework_type_{0};
  int model_type_{0};
  int device_type_{0};
};

class Engine {
 public:
  Engine(int block_idx,
         lite::cpp::BlockDesc *block_desc,
         const std::vector<std::string> &input_names,
         const std::vector<std::string> &output_names,
         lite::Scope *scope)
      : block_idx_(block_idx),
        block_desc_(block_desc),
        input_names_(input_names),
        output_names_(output_names),
        scope_(scope) {}

  int Build();
  int Run();

 private:
  Engine(const Engine &) = delete;
  int CreateDeviceProgram();
  int BuildDeviceProgram();
  int RunDeviceProgram();
  int CreateOriginProgram();
  int BuildOriginProgram();
  int RunOriginProgram();
  bool InputShapeChanged();

 private:
  int block_idx_;
  lite::cpp::BlockDesc *block_desc_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  lite::Scope *scope_{nullptr};
  // SUCCESS: device program build successed. FAILED: device program build
  // failed. REBUILD_WHEN_SHAPE_CHANGED: device program build successed but need
  // to rebuild when input shape changed.
  int build_device_program_status_{0};
  std::vector<lite::DDim> origin_idims_;
  std::vector<lite::DDim> origin_odims_;
  std::vector<lite::Tensor *> origin_itensors_;
  std::vector<lite::Tensor *> origin_otensors_;
  std::string model_name_;
  hiai::AiContext model_context_;
  std::vector<int64_t> device_idatasizes_;
  std::vector<int64_t> device_odatasizes_;
  std::vector<std::shared_ptr<hiai::AiTensor>> device_itensors_;
  std::vector<std::shared_ptr<hiai::AiTensor>> device_otensors_;
  std::unique_ptr<hiai::AiModelMngerClient> device_program_{nullptr};
  std::vector<Instruction> origin_program_;
};

}  // namespace npu
}  // namespace lite
}  // namespace paddle
