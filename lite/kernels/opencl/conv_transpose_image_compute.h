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

#include <memory>
#include <string>
#include <vector>

#include "lite/backends/opencl/cl_half.h"
#include "lite/core/kernel.h"
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

class ConvTransposeImageCompute : public KernelLite<TARGET(kOpenCL),
                                                    PRECISION(kFP16),
                                                    DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ConvParam;
  using kernel_t = void (ConvTransposeImageCompute::*)();

  void PrepareForRun() override;

  void ReInitWhenNeeded() override;

  void Run() override;

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_names_[0];
    ch->global_work_size = ch->NDRangeToStr(global_work_size_);
    ch->local_work_size = ch->NDRangeToStr(local_work_size_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  void Conv2dTranspose();
#ifdef LITE_WITH_LOG
  void PrintConvInfo();
#endif
  void SetGlobalWorkSize();
  void SetArgs();

  param_t* conv_param_{nullptr};

  std::vector<std::string> kernel_func_names_{};
  std::vector<std::string> kernel_func_paths_{};
  std::vector<std::string> build_options_{};
  std::string time_stamp_{GetTimeStamp()};

  std::unique_ptr<Tensor> filter_gpu_image_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_image_{nullptr};
  std::unique_ptr<Tensor> tensor_hold_filter_image_{nullptr};
  std::unique_ptr<Tensor> tensor_hold_bias_image_{nullptr};

  cl::Image2D* input_image_p_{nullptr};
  cl::Image2D* filter_image_p_{nullptr};
  cl::Image2D* bias_image_p_{nullptr};
  cl::Image2D* output_image_p_{nullptr};

  int stride_h_{-1};
  int stride_w_{-1};

  int dilation_h_{-1};
  int dilation_w_{-1};

  int pad_up_{-1};
  int pad_down_{-1};
  int pad_left_{-1};
  int pad_right_{-1};

  int groups_{-1};
  bool relu_fused_{false};
  bool has_bias_{false};

  int input_tensor_n_{-1};
  int input_tensor_c_{-1};
  int input_tensor_h_{-1};
  int input_tensor_w_{-1};
  int input_image_h_{-1};
  int input_image_w_{-1};

  int output_tensor_n_{-1};
  int output_tensor_c_{-1};
  int output_tensor_h_{-1};
  int output_tensor_w_{-1};
  int output_image_h_{-1};
  int output_image_w_{-1};

  int filter_tensor_n_{-1};
  int filter_tensor_c_{-1};
  int filter_tensor_h_{-1};
  int filter_tensor_w_{-1};
  int filter_image_h_{-1};
  int filter_image_w_{-1};

  bool is_group_conv_{false};
  DDim last_input_dims_{};
  bool is_first_epoch_for_run_{true};

  cl::Kernel kernel_;
  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  cl::NDRange local_work_size_{cl::NullRange};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
