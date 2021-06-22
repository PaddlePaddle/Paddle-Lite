// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "graph/tensor.h"
#include "lite/backends/huawei_ascend_npu/device.h"
#include "lite/core/kernel.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/core/subgraph_engine_base.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace huawei_ascend_npu {

using TensorDesc = paddle::lite::huawei_ascend_npu::TensorDesc;
using AclModelClient = paddle::lite::huawei_ascend_npu::AclModelClient;

class DeviceProgram {
 public:
  DeviceProgram() {}
  ~DeviceProgram() {}
  std::string GenerateModelName(
      const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names,
      const std::vector<std::vector<int64_t>>& origin_idims);
  bool LoadFromCacheFile(const std::vector<std::string>& input_names,
                         const std::vector<std::string>& output_names,
                         const std::vector<std::vector<int64_t>>& origin_idims,
                         const std::string& model_cache_dir,
                         const int device_id);
  bool BuildGraphAndCacheToFile(
      RuntimeProgram* origin_program,
      const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names,
      const std::vector<std::vector<int64_t>>& origin_idims,
      const std::vector<Tensor*>& origin_otensors,
      const std::string& model_cache_dir,
      const int device_id);
  bool ShareBufferWithOriginTensors(
      const std::vector<std::string>& input_names,
      const std::vector<std::string>& output_names,
      std::vector<Tensor*>* origin_itensors,
      std::vector<Tensor*>* origin_otensors,
      std::vector<std::shared_ptr<ge::Tensor>>* device_itensors,
      std::vector<std::shared_ptr<ge::Tensor>>* device_otensors);
  bool SharedBufferWithOutputTensors(
      const std::vector<std::string>& output_names,
      std::vector<Tensor*>* origin_otensors,
      std::vector<std::shared_ptr<ge::Tensor>>* device_otensors);
  bool ZeroCopyRun(std::vector<std::shared_ptr<ge::Tensor>>* device_itensors,
                   std::vector<std::shared_ptr<ge::Tensor>>* device_otensors);

 public:
  std::string model_name_{""};
  std::shared_ptr<AclModelClient> model_client_{nullptr};
  std::vector<std::vector<int64_t>> origin_odims_;
  std::vector<PrecisionType> origin_otypes_;
  std::vector<TensorDesc> device_idims_{};
  std::vector<TensorDesc> device_odims_{};
};

class SubgraphEngine : public subgraph::SubgraphEngineBase {
 public:
  SubgraphEngine(KernelContext* ctx,
                 int block_idx,
                 const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
                 Scope* exec_scope,
                 const std::vector<std::string>& input_names,
                 const std::vector<std::string>& output_names)
      : subgraph::SubgraphEngineBase(ctx,
                                     block_idx,
                                     program_desc,
                                     exec_scope,
                                     input_names,
                                     output_names) {
    lite::huawei_ascend_npu::Device::Global().RegisterDeviceResource();
  }

  virtual ~SubgraphEngine() {
    lite::huawei_ascend_npu::Device::Global().UnRegisterDeviceResource();
  }

 protected:
  bool PrepareWorkspaceForDeviceProgram() override;
  bool BuildDeviceProgram() override;
  bool LaunchDeviceProgram() override;

 private:
  std::vector<std::shared_ptr<ge::Tensor>> device_itensors_{};
  std::vector<std::shared_ptr<ge::Tensor>> device_otensors_{};
  std::map<std::vector<std::vector<int64_t>>, std::shared_ptr<DeviceProgram>>
      device_programs_;
};

class SubgraphCompute
    : public KernelLite<TARGET(kHuaweiAscendNPU), PRECISION(kAny)> {
 public:
  using param_t = operators::SubgraphParam;
  void PrepareForRun() override;
  void Run() override;
  virtual ~SubgraphCompute() = default;

 private:
  std::unique_ptr<SubgraphEngine> engine_;
};

}  // namespace huawei_ascend_npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
