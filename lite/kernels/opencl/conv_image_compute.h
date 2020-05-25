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

#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {
class ConvImageCompute : public KernelLite<TARGET(kOpenCL),
                                           PRECISION(kFP16),
                                           DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ConvParam;
  using kernel_t = void (ConvImageCompute::*)(bool);

  void PrepareForRun() override;

  void Run() override;
  double Turn(int times = 5);

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_names_[0];
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  void Conv2d1x1opt(bool is_turn = false);
  void Conv2d3x3(bool is_turn = false);
  void Conv2d3x3opt(bool is_turn = false);
  void Conv2d5x5(bool is_turn = false);
  void Conv2d5x5opt(bool is_turn = false);
  void Conv2d7x7(bool is_turn = false);
  void Conv2d7x7opt(bool is_turn = false);
  void DepthwiseConv2d3x3s1(bool is_turn = false);
  void DepthwiseConv2d3x3(bool is_turn = false);
  void DepthwiseConv2d(bool is_turn = false);

  kernel_t impl_;
  std::vector<std::string> kernel_func_names_{};
  std::vector<std::string> kernel_func_paths_{};
  std::vector<std::string> build_options_{};
  std::string time_stamp_{GetTimeStamp()};

  std::unique_ptr<Tensor> filter_gpu_image_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_image_{nullptr};
  std::unique_ptr<Tensor> tensor_hold_filter_image_{nullptr};
  std::unique_ptr<Tensor> tensor_hold_bias_image_{nullptr};
  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  int c_blk_ = 1;
  int w_blk_ = 1;
  int nh_blk_ = 1;

  int default_c_blk_ = 1;
  int default_w_blk_ = 1;
  int default_nh_blk_ = 1;

  cl::Kernel kernel_;
  cl::NDRange local_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  bool use_lws_{true};
  bool use_turn_{false};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
