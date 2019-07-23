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
#include <vector>
#include "ai_ddk_lib/include/HiAiModelManagerService.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {

class GraphCompute : public KernelLite<TARGET(kNPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::GraphParam;

  void PrepareForRun() override;

  void Run() override;

  virtual ~GraphCompute() = default;

  bool input_dims_changed() const;

 private:
  DDim input_dims_;
  hiai::AiModelMngerClient* exec_;
  std::vector<hiai::TensorDimension> npu_idims_;
  std::vector<hiai::TensorDimension> npu_odims_;

  std::vector<std::shared_ptr<hiai::AiTensor>> npu_itensors_;
  std::vector<std::shared_ptr<hiai::AiTensor>> npu_otensors_;

  // TODO(TJ): find better place
  hiai::AiContext npu_context_;
};

}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
