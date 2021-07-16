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
#include <cmath>
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class BatchNormComputeImage2D : public KernelLite<TARGET(kOpenCL),
                                                  PRECISION(kFP16),
                                                  DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::BatchNormParam;

  std::string doc() const override {
    return "BatchNorm using cl::Image2D, kFP16";
  }

  void PrepareForRun() override {
    batch_norm_param_ = param_.get_mutable<param_t>();
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/batch_norm_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());

    auto& out_dims = batch_norm_param_->y->dims();
    out_w_ = out_dims[3];

    scale_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
    bias_gpu_image_ = std::unique_ptr<Tensor>(new Tensor);
    tensor_hold_scale_image_ = std::unique_ptr<Tensor>(new Tensor);
    tensor_hold_bias_image_ = std::unique_ptr<Tensor>(new Tensor);

    CLImageConverterFolder scale_bias_converter;
    const DDim& scale_image_dims = scale_bias_converter.InitImageDimInfoWith(
        batch_norm_param_->scale->dims());
    const DDim& bias_image_dims = scale_bias_converter.InitImageDimInfoWith(
        batch_norm_param_->bias->dims());
    tensor_hold_scale_image_->Resize(
        {1, scale_image_dims[0], scale_image_dims[1], 4});
    tensor_hold_bias_image_->Resize(
        {1, bias_image_dims[0], bias_image_dims[1], 4});
    auto* scale_image_data = MUTABLE_DATA_CPU(tensor_hold_scale_image_);
    auto* bias_image_data = MUTABLE_DATA_CPU(tensor_hold_bias_image_);
    auto* scale_cpu_data = batch_norm_param_->scale->data<float>();
    auto* bias_cpu_data = batch_norm_param_->bias->data<float>();
    auto* mean_cpu_data = batch_norm_param_->mean->data<float>();
    auto* variance_cpu_data = batch_norm_param_->variance->data<float>();

    int element_num = batch_norm_param_->scale->dims().production();
    std::vector<float> new_scale_data(element_num);
    std::vector<float> new_bias_data(element_num);
    for (int i = 0; i < element_num; i++) {
      float inv_scale =
          1.f / (std::sqrt(variance_cpu_data[i] + batch_norm_param_->epsilon));
      float new_bias =
          bias_cpu_data[i] - inv_scale * scale_cpu_data[i] * mean_cpu_data[i];
      float new_scale = inv_scale * scale_cpu_data[i];
      new_scale_data[i] = new_scale;
      new_bias_data[i] = new_bias;
    }
    scale_bias_converter.NCHWToImage(new_scale_data.data(),
                                     scale_image_data,
                                     batch_norm_param_->scale->dims());
    scale_bias_converter.NCHWToImage(
        new_bias_data.data(), bias_image_data, batch_norm_param_->bias->dims());

    MUTABLE_DATA_GPU(scale_gpu_image_,
                     scale_image_dims[0],
                     scale_image_dims[1],
                     scale_image_data);
    MUTABLE_DATA_GPU(bias_gpu_image_,
                     bias_image_dims[0],
                     bias_image_dims[1],
                     bias_image_data);
  }

  void ReInitWhenNeeded() override {
    auto x_dims = batch_norm_param_->x->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute image shape
      CLImageConverterDefault default_convertor;
      out_img_shape_ =
          default_convertor.InitImageDimInfoWith(batch_norm_param_->y->dims());

      // compute global work size
      GetGlobalWorkSize();
    }
  }

  void GetGlobalWorkSize() {
    global_work_size_ =
        cl::NDRange{static_cast<cl::size_type>(out_img_shape_[0]),
                    static_cast<cl::size_type>(out_img_shape_[1])};
  }

  void Run() override {
    auto* x_img = GET_DATA_GPU(batch_norm_param_->x);
    auto* out_img = MUTABLE_DATA_GPU(
        batch_norm_param_->y, out_img_shape_[0], out_img_shape_[1], nullptr);
    auto* scale_img = GET_DATA_GPU(scale_gpu_image_);
    auto* bias_img = GET_DATA_GPU(bias_gpu_image_);

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto kernel = kernel_;
    cl_int status;
    status = kernel.setArg(0, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *scale_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, *bias_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, out_w_);
    CL_CHECK_FATAL(status);

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  std::string kernel_func_name_{"batch_norm"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};

  param_t* batch_norm_param_{nullptr};
  cl::Kernel kernel_;
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  DDim out_img_shape_ = DDim(std::vector<DDim::value_type>(
      {static_cast<DDim::value_type>(1), static_cast<DDim::value_type>(1)}));
  cl::NDRange global_work_size_ = cl::NDRange{
      static_cast<size_t>(1), static_cast<size_t>(1), static_cast<size_t>(1)};
  int out_w_{0};
  std::unique_ptr<Tensor> scale_gpu_image_{nullptr};
  std::unique_ptr<Tensor> bias_gpu_image_{nullptr};
  std::unique_ptr<Tensor> tensor_hold_scale_image_{nullptr};
  std::unique_ptr<Tensor> tensor_hold_bias_image_{nullptr};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(batch_norm,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::BatchNormComputeImage2D,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("ReserveSpace", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

REGISTER_LITE_KERNEL(sync_batch_norm,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     paddle::lite::kernels::opencl::BatchNormComputeImage2D,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Variance", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindOutput("MeanOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("ReserveSpace", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("VarianceOut", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
