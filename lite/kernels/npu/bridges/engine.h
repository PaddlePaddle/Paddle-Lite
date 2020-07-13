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

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/program.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {

class Engine {
 public:
  Engine(KernelContext *ctx,
         int block_idx,
         cpp::BlockDesc *block_desc,
         const std::vector<std::string> &input_names,
         const std::vector<std::string> &output_names,
         lite::Scope *scope);
  virtual ~Engine() = default;

  virtual bool Run();

 private:
  Engine(const Engine &) = delete;

 protected:
  virtual bool PrepareWorkspaceForOriginProgram();
  virtual bool BuildOriginProgram();
  virtual bool LaunchOriginProgram();

  virtual bool PrepareWorkspaceForDeviceProgram();
  virtual bool BuildDeviceProgram();
  virtual bool LaunchDeviceProgram();

  virtual bool InputShapeChanged();

  KernelContext *ctx_{nullptr};
  int block_idx_{-1};
  cpp::BlockDesc *block_desc_{nullptr};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  Scope *scope_{nullptr};
  bool is_first_epoch_{true};
  std::vector<std::vector<int64_t>> origin_idims_;
  std::vector<Tensor *> origin_itensors_;
  std::vector<Tensor *> origin_otensors_;
  std::vector<Instruction> origin_program_;
};

}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
