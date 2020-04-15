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

#include <graph/tensor.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/hw_ascend_npu/runtime.h"
#include "lite/core/kernel.h"
#include "lite/kernels/npu/bridges/engine.h"
#include "lite/kernels/npu/bridges/registry.h"
using HWAscendNPURuntime = paddle::lite::hw_ascend_npu::HWAscendNPURuntime;
using TensorDesc = paddle::lite::hw_ascend_npu::TensorDesc;
namespace paddle {
namespace lite {
namespace kernels {
namespace hw_ascend_npu {

class SubgraphEngine : public subgraph::Engine {
 public:
  SubgraphEngine(KernelContext *ctx,
                 int block_idx,
                 cpp::BlockDesc *block_desc,
                 const std::vector<std::string> &input_names,
                 const std::vector<std::string> &output_names,
                 Scope *scope)
      : subgraph::Engine(
            ctx, block_idx, block_desc, input_names, output_names, scope) {}

  struct device_program_t {
    explicit device_program_t(std::shared_ptr<HWAscendNPURuntime> _client)
        : client(_client) {}
    std::shared_ptr<HWAscendNPURuntime> client{nullptr};
    std::vector<DDim> origin_idims{};
    std::vector<DDim> origin_odims{};
    std::vector<TensorDesc> device_idims{};
    std::vector<TensorDesc> device_odims{};
  };

 protected:
  int BuildDeviceProgram() override;
  int LaunchDeviceProgram() override;
  bool InputShapeChanged() override;

  std::vector<std::vector<int64_t>> inputs_shape_{};
  std::map<std::vector<std::vector<int64_t>>, std::shared_ptr<device_program_t>>
      device_program_map_{};
  std::vector<std::string> device_inames_{};
  std::vector<std::string> device_onames_{};
};

class SubgraphCompute
    : public KernelLite<TARGET(kHWAscendNPU), PRECISION(kAny)> {
 public:
  using param_t = operators::SubgraphParam;

  void PrepareForRun() override;

  void Run() override;

  virtual ~SubgraphCompute() = default;

 private:
  std::unique_ptr<SubgraphEngine> engine_;
};

}  // namespace hw_ascend_npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
