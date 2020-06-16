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
#include <string>
#include "compute_param.h"  // NOLINT
#include "paddle_place.h"   // NOLINT

namespace paddle {
namespace lite_api {

// now ComputeEngine only support Target = Arm
template <TargetType Type>
class LITE_API ComputeEngine {
 public:
  ComputeEngine() = default;
  bool CreateOperator(const char* op_type,
                      PrecisionType precision = PRECISION(kFloat),
                      DataLayoutType layout = DATALAYOUT(kNCHW)) {}
  void SetParam(ParamBase* param) {}
  void Launch() {}
  ~ComputeEngine() = default;

 private:
  void* instruction_;
  void* param_;
};

template <>
class LITE_API ComputeEngine<TARGET(kARM)> {
 public:
  ComputeEngine() = default;
  static void env_init(PowerMode power_mode, int threads);
  bool CreateOperator(const char* op_type,
                      PrecisionType precision = PRECISION(kFloat),
                      DataLayoutType layout = DATALAYOUT(kNCHW));
  void SetParam(ParamBase* param);
  void Launch();
  ~ComputeEngine();

 private:
  void* instruction_{nullptr};
  void* param_{nullptr};
};

}  // namespace lite_api
}  // namespace paddle
