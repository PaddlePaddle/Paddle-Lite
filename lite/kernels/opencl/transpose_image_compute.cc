// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/log/logging.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

#undef LITE_WITH_LOG

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class TransposeComputeFloatImage
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::TransposeParam;

  void PrepareForRun() override {
    transpose_param_ = param_.get_mutable<param_t>();
    axis_ = transpose_param_->axis;
    auto x = transpose_param_->x;
    x_tensor_dims_ = x->dims();
    auto output = transpose_param_->output;
    output_tensor_dims_ = output->dims();
    auto output_image_shape = InitImageDimInfoWith(output_tensor_dims_);
    output_image_h_ = output_image_shape.at("height");
    output_image_w_ = output_image_shape.at("width");
    VLOG(4) << "x_tensor_dims_: " << x_tensor_dims_;

    if (axis_.size() == 3) {
      VLOG(4) << "Extend CHW to 1CHW";
      axis_.insert(axis_.begin(), 0);  // extend batch dim is 1
      for (int i = 1; i < axis_.size(); ++i) {
        axis_[i]++;
      }
    }
    if (axis_.size() == 4) {
      std::vector<int> tmp = axis_;
      sort(tmp.begin(), tmp.end());
      if (tmp == std::vector<int>({0, 1, 2, 3}) &&
          axis_ != std::vector<int>({0, 1, 2, 3})) {
        kernel_func_name_ = "transpose_4d_perm";
        for (int i = 0; i < axis_.size(); ++i) {
          kernel_func_name_ += to_string(axis_[i]);
        }
        kernel_path_ =
            "image/transpose_fixb" + to_string(axis_[0] + 1) + "_kernel.cl";
      } else {
        LOG(FATAL) << "Unsupported axis permutation for current lite OpenCL "
                      "kernel! ";
      }
    } else if (axis_.size() == 2) {
      std::vector<int> tmp = axis_;
      sort(tmp.begin(), tmp.end());
      if (tmp == std::vector<int>({0, 1}) &&
          axis_ != std::vector<int>({0, 1})) {
        kernel_func_name_ = "transpose_2d";
        kernel_path_ = "image/transpose_fixb1_kernel.cl";
      } else {
        LOG(FATAL) << "Unsupported axis permutation for current lite OpenCL "
                      "kernel! ";
      }
    } else {
      LOG(FATAL) << "Unsupported axis permutation for current lite OpenCL "
                    "kernel! ";
    }

    if (output_tensor_dims_.size() == 4) {
      output_tensor_n_ = output_tensor_dims_[0];
      output_tensor_c_ = output_tensor_dims_[1];
      output_tensor_h_ = output_tensor_dims_[2];
      output_tensor_w_ = output_tensor_dims_[3];
      x_tensor_w_ = x_tensor_dims_[3];
      x_tensor_h_ = x_tensor_dims_[2];
    } else if (output_tensor_dims_.size() == 3) {
      output_tensor_c_ = output_tensor_dims_[0];
      output_tensor_h_ = output_tensor_dims_[1];
      output_tensor_w_ = output_tensor_dims_[2];
      x_tensor_w_ = x_tensor_dims_[2];
      x_tensor_h_ = x_tensor_dims_[1];
    } else if (output_tensor_dims_.size() == 2) {
      output_tensor_h_ = output_tensor_dims_[0];
      output_tensor_w_ = output_tensor_dims_[1];
      x_tensor_w_ = x_tensor_dims_[1];
    }

    auto& context = ctx_->As<OpenCLContext>();
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
    context.cl_context()->AddKernel(
        kernel_func_name_, kernel_path_, build_options_, time_stamp_);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

  void GetGlobalWorkSize() {
    const std::vector<size_t>& ws =
        DefaultGlobalWorkSize(output_tensor_dims_,
                              DDim(std::vector<DDim::value_type>{
                                  static_cast<int64_t>(output_image_w_),
                                  static_cast<int64_t>(output_image_h_)}));
    global_work_size_ = cl::NDRange{static_cast<cl::size_type>(ws[0]),
                                    static_cast<cl::size_type>(ws[1]),
                                    static_cast<cl::size_type>(ws[2])};
  }

  void Run() override {
    auto* x_image = GET_DATA_GPU(transpose_param_->x);
    auto* output_image = MUTABLE_DATA_GPU(
        transpose_param_->output, output_image_w_, output_image_h_, nullptr);

    auto& context = ctx_->As<OpenCLContext>();
    auto kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_image);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *output_image);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, output_tensor_c_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, output_tensor_h_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, output_tensor_w_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, x_tensor_w_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(6, x_tensor_h_);
    CL_CHECK_FATAL(status);

    GetGlobalWorkSize();
    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

 private:
  std::string kernel_func_name_{""};
  std::string kernel_path_{""};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};

  param_t* transpose_param_{nullptr};
  std::unique_ptr<Tensor> output_tensor_idxs_t_{nullptr};
  cl::Buffer* output_tensor_idxs_data_;

  std::vector<int> axis_;
  DDim x_tensor_dims_{};
  int x_tensor_w_{1};
  int x_tensor_h_{1};
  DDim output_tensor_dims_{};
  int output_tensor_n_{1};
  int output_tensor_c_{1};
  int output_tensor_h_{1};
  int output_tensor_w_{1};
  int output_image_h_{1};
  int output_image_w_{1};

  cl::NDRange global_work_size_;
  cl::Kernel kernel_;

  // transpose_general_buffer
  std::unique_ptr<KernelBase> im2buf_kernel_;
  std::unique_ptr<KernelBase> buf2im_kernel_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(transpose,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::TransposeComputeFloatImage,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(transpose2,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::TransposeComputeFloatImage,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

#define LITE_WITH_LOG
