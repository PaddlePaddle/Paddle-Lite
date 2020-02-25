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

#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class NearestInterpComputeImageDefault
    : public KernelLite<TARGET(kOpenCL),
                        PRECISION(kFP16),
                        DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::InterpolateParam;

  std::string doc() const override {
    return "NearestInterp using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, "image/nearest_interp_kernel.cl", build_options_);
  }

  void Run() override {
    auto& param = *param_.get_mutable<param_t>();
    const auto& x_dims = param.X->dims();
    const auto& y_dims = param.Out->dims();
    auto* x_buf =
        param.X->data<half_t,
                      cl::Image2D>();  // use half_t represents half float
    auto out_image_shape = InitImageDimInfoWith(y_dims);
    auto* out_buf = param.Out->mutable_data<half_t, cl::Image2D>(  // use half_t
        // represents half float
        out_image_shape["width"],
        out_image_shape["height"]);

    float scale_h = y_dims[2] / x_dims[2];
    float scale_w = y_dims[3] / x_dims[3];
    int in_dims_h = x_dims[2];
    int out_dims_h = y_dims[2];
    int in_dims_w = x_dims[3];
    int out_dims_w = y_dims[3];

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    int arg_idx = 0;
    cl_int status = kernel.setArg(arg_idx, *x_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(scale_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const float>(scale_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims_h));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(in_dims_w));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(++arg_idx, static_cast<const int>(out_dims_w));
    CL_CHECK_FATAL(status);

    VLOG(4) << TargetToStr(param.X->target());
    VLOG(4) << TargetToStr(param.Out->target());
    VLOG(4) << "out_image_shape(w,h):" << out_image_shape["width"] << " "
            << out_image_shape["height"];
    VLOG(4) << "x_dims[" << x_dims.size() << "D]:" << x_dims[0] << " "
            << x_dims[1] << " " << x_dims[2] << " " << x_dims[3];
    VLOG(4) << "y_dims[" << y_dims.size() << "D]:" << y_dims[0] << " "
            << y_dims[1] << " " << y_dims[2] << " " << y_dims[3];

    const std::vector<size_t>& default_work_size =
        DefaultWorkSize(y_dims,
                        DDim(std::vector<DDim::value_type>{
                            static_cast<int64_t>(out_image_shape["width"]),
                            static_cast<int64_t>(out_image_shape["height"])}));
    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(default_work_size.data()[0]),
                    static_cast<cl::size_type>(default_work_size.data()[1]),
                    static_cast<cl::size_type>(default_work_size.data()[2])};
    status = context.cl_context()->GetCommandQueue().enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global_work_size,
        cl::NullRange,
        nullptr,
        event_.get());
    CL_CHECK_FATAL(status);
    // TODO(ysh329): io_copy(device->host) jammed if emplace to `cl_wait_list`
    // context.cl_wait_list()->emplace(out_buf, event_);
    context.cl_context()->GetCommandQueue().finish();

    auto tensor_mean_cl = [](const Tensor* in,
                             PrecisionType ptype,
                             std::string name = "inst") -> double {
      if (!in->data<int8_t>()) {
        return -99999;
      }
      double sum = 0.;
      // profile opencl
      switch (ptype) {
        case PRECISION(kFloat): {
          paddle::lite::CLImageConverterDefault default_convertor;
          DDim out_image_shape =
              default_convertor.InitImageDimInfoWith(in->dims());
          int out_image_width = out_image_shape[0];
          int out_image_height = out_image_shape[1];

          const size_t cl_image2d_row_pitch{0};
          const size_t cl_image2d_slice_pitch{0};
          VLOG(4) << "out_image_shape: " << out_image_shape[0] << "  "
                  << out_image_shape[1];
          std::vector<uint16_t> out_image_v(out_image_shape.production() *
                                            4);  // 4 :RGBA
          std::vector<float> output_v(in->dims().production());
          auto* indata = in->data<float, cl::Image2D>();
          VLOG(4) << "indata addr: " << indata;
          if (indata == nullptr) {
            return -1;
          }
          TargetWrapperCL::ImgcpySync(out_image_v.data(),
                                      in->data<uint16_t, cl::Image2D>(),
                                      out_image_width,
                                      out_image_height,
                                      cl_image2d_row_pitch,
                                      cl_image2d_slice_pitch,
                                      IoDirection::DtoH);
          // LOG(INFO) << "out_image_v: ";
          // stride_print(out_image_v.size(), out_image_v.data());
          default_convertor.ImageToNCHW(
              out_image_v.data(), output_v.data(), out_image_shape, in->dims());
          // LOG(INFO) << "output_v: ";
          // stride_print(output_v.size(), output_v.data());
          for (size_t i = 0; i < output_v.size(); i++) {
            sum += output_v[i];
          }

          return sum / in->numel();
        }

        default:
          LOG(INFO) << "opencl unsupport data type: " << PrecisionToStr(ptype);
          return 0.;
      }
    };
    double mean =
        tensor_mean_cl(param.Out, PrecisionType::kFloat, "nearinterp");
    LOG(INFO) << "nearest out mean: " << mean;
  }

 private:
  std::string kernel_func_name_{"nearest_interp"};
  std::string build_options_{" -DCL_DTYPE_half"};
  std::shared_ptr<cl::Event> event_{new cl::Event};
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    nearest_interp,
    kOpenCL,
    kFP16,
    kImageDefault,
    paddle::lite::kernels::opencl::NearestInterpComputeImageDefault,
    ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
