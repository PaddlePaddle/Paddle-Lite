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
#include "lite/core/kernel.h"
#include "lite/operators/op_params.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

template <PrecisionType Ptype, DataLayoutType layout>
class ConcatCompute : public KernelLite<TARGET(kOpenCL), Ptype, layout> {
 public:
  using param_t = operators::ConcatParam;

  void PrepareForRun() override;

  void Run() override;

  std::string doc();  // override;

  // protected:
  // void UpdateParams();

  int axis_size_ = 1;
  int post_size_ = 1;
  int pre_size_ = 1;
  int axis_ = 1;
  param_t* concat_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{"-DCL_DTYPE_float"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
