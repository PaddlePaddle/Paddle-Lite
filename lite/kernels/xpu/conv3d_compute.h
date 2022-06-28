// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

template <typename TGEMM,
          typename TW,
          typename DX,
          typename DY,
          PrecisionType PType>
class Conv3DCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::ConvParam;
  void PrepareForRun() override;

  void Run() override;

  virtual ~Conv3DCompute() = default;

 private:
  XPUQuantData xpu_quant_filter_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
