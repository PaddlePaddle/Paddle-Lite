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

#include <memory>
#include <string>
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_image_converter.h"
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

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {
class BilinearInterpImageCompute
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::InterpolateParam;

  std::string doc() const override {
    return "BilinearInterp using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    bilinear_interp_param_ = param_.get_mutable<param_t>();

    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/bilinear_interp_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
  }

  inline std::vector<int> get_new_shape(
      std::vector<const lite::Tensor*> list_new_shape_tensor) {
    // get tensor from
    std::vector<int> vec_new_shape;
    for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
      auto tensor = list_new_shape_tensor[i];
      vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
    }
    return vec_new_shape;
  }

  template <typename T>
  inline std::vector<T> get_new_data_from_tensor(
      const Tensor* new_data_tensor) {
    std::vector<T> vec_new_data;
    auto* new_data = new_data_tensor->data<T>();
    lite::Tensor cpu_starts_tensor;
    vec_new_data = std::vector<T>(
        new_data, new_data + new_data_tensor->dims().production());
    return vec_new_data;
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    auto* x = bilinear_interp_param_->X;
    auto* out = bilinear_interp_param_->Out;

    // init param
    auto size_tensor = bilinear_interp_param_->SizeTensor;
    int out_width = out->dims()[3];
    int out_height = out->dims()[2];
    if (bilinear_interp_param_->out_w > 0 &&
        bilinear_interp_param_->out_h > 0) {
      out_width = bilinear_interp_param_->out_w;
      out_height = bilinear_interp_param_->out_h;
    }
    float scale = bilinear_interp_param_->scale;

    // update out dims
    int in_h = x->dims()[2];
    int in_w = x->dims()[3];
    if (size_tensor.size() > 0) {
      auto new_size = get_new_shape(size_tensor);
      out_height = new_size[0];
      out_width = new_size[1];
    } else {
      auto scale_tensor = bilinear_interp_param_->Scale;
      if (scale_tensor != nullptr) {
        auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
        scale = scale_data[0];
      }
      if (scale > 0) {
        out_height = static_cast<int>(in_h * scale);
        out_width = static_cast<int>(in_w * scale);
      }
      auto out_size = bilinear_interp_param_->OutSize;
      if (out_size != nullptr) {
        auto out_size_data = get_new_data_from_tensor<int>(out_size);
        out_height = out_size_data[0];
        out_width = out_size_data[1];
      }
    }

    int num_cout = x->dims()[0];
    int c_cout = x->dims()[1];
    out->Resize({num_cout, c_cout, out_height, out_width});

    auto out_dims = out->dims();
    auto in_dims = x->dims();
    float scale_h = 0.0;
    float scale_w = 0.0;

    if (bilinear_interp_param_->align_corners) {
      scale_h = (in_dims[2] - 1.0f) / (out_dims[2] - 1.0f);
      scale_w = (in_dims[3] - 1.0f) / (out_dims[3] - 1.0f);
    } else {
      scale_h = in_dims[2] / static_cast<float>(out_dims[2]);
      scale_w = in_dims[3] / static_cast<float>(out_dims[3]);
    }
    float align_delta = 0.0f;
    if (!bilinear_interp_param_->align_corners &&
        bilinear_interp_param_->align_mode == 0) {
      align_delta = 0.5f;
    }

    int out_h = out_dims[2];
    int out_w = out_dims[3];

#ifdef LITE_WITH_LOG
    VLOG(4) << "x->target():" << TargetToStr(x->target());
    VLOG(4) << "out->target():" << TargetToStr(out->target());
    VLOG(4) << "x->dims():" << in_dims;
    VLOG(4) << "out->dims():" << out_dims;
#endif

    auto out_image_shape = InitImageDimInfoWith(out_dims);
    auto* x_img = GET_DATA_GPU(x);
    auto* out_img = MUTABLE_DATA_GPU(
        out, out_image_shape["width"], out_image_shape["height"], nullptr);

#ifdef LITE_WITH_LOG
    // VLOG(4) << "x_image: " << x_img;
    // VLOG(4) << "out_image: " << out_img;
    VLOG(4) << "out_image_shape[w,h]: " << out_image_shape["width"] << " "
            << out_image_shape["height"];

    VLOG(4) << "scale_h: " << scale_h << ", scale_w: " << scale_w
            << ", align_delta: " << align_delta;
    VLOG(4) << "in_h: " << in_h << ", in_w: " << in_w;
    VLOG(4) << "out_h: " << out_h << ", out_w: " << out_w;
#endif

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    auto default_work_size = DefaultGlobalWorkSize(
        out_dims,
        DDim(std::vector<DDim::value_type>{
            static_cast<int64_t>(out_image_shape["width"]),
            static_cast<int64_t>(out_image_shape["height"])}));
#ifdef LITE_WITH_LOG
    VLOG(4) << "default_work_size: " << default_work_size[0] << ", "
            << default_work_size[1] << ", " << default_work_size[2];
#endif
    cl_int status = kernel.setArg(arg_idx++, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, scale_h);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, scale_w);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, align_delta);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, in_h);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, in_w);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, out_h);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, out_w);
    CL_CHECK_FATAL(status);

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(default_work_size[0]),
                    static_cast<cl::size_type>(default_work_size[1]),
                    static_cast<cl::size_type>(default_work_size[2])};

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
#ifdef LITE_WITH_LOG
    VLOG(4) << "global_work_size:[2D]:" << global_work_size[0] << " "
            << global_work_size[1] << " " << global_work_size[2];
#endif
  }

 protected:
  param_t* bilinear_interp_param_{nullptr};
  std::string kernel_func_name_{"bilinear_interp"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(bilinear_interp,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::BilinearInterpImageCompute,
                     ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();

REGISTER_LITE_KERNEL(bilinear_interp_v2,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::BilinearInterpImageCompute,
                     ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
