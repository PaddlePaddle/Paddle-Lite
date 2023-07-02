// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>
#include "lite/core/kernel.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
template <PrecisionType PType>
class FusedAttentionCompute : public KernelLite<TARGET(kARM), PType> {
 public:
  using param_t = operators::FusedAttentionParam;

  FusedAttentionCompute() = default;

  virtual void ReInitWhenNeeded();
  virtual void PrepareForRun();
  virtual void Run();

  virtual ~FusedAttentionCompute() = default;

 private:
  DDim last_shape_;
  operators::ActivationParam act_param_;
  int fc_m_;
  int fc_k_;
  int fc_n_;
  DDim fc_dims_;

  std::vector<int> reshape_shape_;
  DDim transpose_out_dim_;

  int fc1_m_;
  int fc1_n_;
  int fc1_k_;
  std::vector<float> fc1_scale_;
  DDim fc1_out_dim_;

  DDim softmax_out_dim_;
  DDim calib1_dims_;

  int fc2_m_;
  int fc2_n_;
  int fc2_k_;
  std::vector<float> fc2_scale_;
  DDim fc2_out_dim_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
