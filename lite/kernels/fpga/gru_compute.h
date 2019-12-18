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
#include <algorithm>
#include "lite/core/kernel.h"

#include "lite/backends/fpga/KD/pes/elementwise_add_pe.hpp"
#include "lite/backends/fpga/KD/pes/fully_connected_pe.hpp"
#include "lite/backends/fpga/KD/pes/gru_pe.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

class GRUCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::GRUParam;

  GRUCompute() = default;

  void PrepareForRun() override;

  void Run() override;

  virtual ~GRUCompute() = default;

 private:
  zynqmp::Tensor pre_output_;
  zynqmp::Tensor pre_bias_;
  zynqmp::Tensor weight_;

  zynqmp::ElementwiseAddPE bias_ew_pe_;
  zynqmp::FullyConnectedPE pre_out_pe_;
  zynqmp::FullyConnectedPE reset_out_pe_;

  zynqmp::GRUPE pe_;
};

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
