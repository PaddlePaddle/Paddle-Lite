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
#include <vector>
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/tensor.h"
#include "lite/operators/op_params.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ConvCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::ConvParam;
  using kernel_t = void (ConvCompute::*)();

  void PrepareForRun() override;

  void Run() override;

 private:
  void GemmlikeConv2d();
  void Conv2d1x1();
  void GemmBatched(cl::Kernel& kernel,
                   const cl::Buffer* x_d,
                   const cl::Buffer* filter_d,
                   const cl::Buffer* bias_d,
                   cl::Buffer* output_d,
                   const int batch_size,
                   const int m,
                   const int n,
                   const int k);
  kernel_t impl_;
  std::unique_ptr<lite::Tensor> col_buffer_{nullptr};
  std::vector<std::string> kernel_func_names_{};
  std::vector<std::string> kernel_func_paths_{};
  std::vector<std::string> build_options_{};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
