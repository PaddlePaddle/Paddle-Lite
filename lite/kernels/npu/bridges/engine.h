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
#include <unordered_map>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/program.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {

typedef struct {
  DDim dims;
  LoD lod;
} Shape;

class Engine {
 public:
  Engine(KernelContext *ctx,
         int block_idx,
         cpp::ProgramDesc *program_desc,
         Scope *exec_scope,
         const std::vector<std::string> &input_names,
         const std::vector<std::string> &output_names,
         const std::vector<std::string> &cached_shapes = {},
         std::string model_cache_dir = "");
  virtual ~Engine() = default;

  virtual int Build();
  virtual int Launch();

 private:
  Engine(const Engine &) = delete;

 protected:
  virtual int BuildDeviceProgram();
  virtual int LaunchDeviceProgram();

  virtual int BuildOriginProgram();
  virtual int LaunchOriginProgram();

  virtual void InitDeviceTensor();
  virtual bool InputShapeChanged();

  KernelContext *ctx_{nullptr};
  int block_idx_{-1};
  cpp::ProgramDesc *program_desc_{nullptr};
  Scope *exec_scope_{nullptr};
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::vector<Shape>, std::vector<Shape>> cached_shapes_;
  // SUCCESS: device program build successed. FAILED: device program build
  // failed. REBUILD_WHEN_SHAPE_CHANGED: device program build successed but need
  // to rebuild when input shape changed.
  int build_device_program_status_{0};
  std::vector<DDim> origin_idims_;
  std::vector<DDim> origin_odims_;
  std::vector<Tensor *> origin_itensors_;
  std::vector<Tensor *> origin_otensors_;
  std::unique_ptr<RuntimeProgram> origin_program_;
  std::string model_cache_dir_{""};
};

}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
