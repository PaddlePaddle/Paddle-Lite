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
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"
#include "lite/backends/opencl/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class ConcatComputeImage : public KernelLite<TARGET(kOpenCL),
                                             PRECISION(kFP16),
                                             DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::ConcatParam;

  void PrepareForRun() override {
    auto& context = ctx_->As<OpenCLContext>();
    concat_param_ = param_.get_mutable<param_t>();
    axis_ = concat_param_->axis;
    auto* axis_tensor = concat_param_->axis_tensor;
    if (axis_tensor != nullptr) {
      auto* axis_tensor_data = axis_tensor->data<int>();
      axis_ = axis_tensor_data[0];
    }
    /*
        if (axis_tensor != nullptr) {
          auto* d_image = DATA_GPU(axis_tensor);
          cl::Image2D* cl_image = static_cast<cl::Image2D*>(d_image);
          size_t cl_image2d_width, cl_image2d_height;
          cl_image->getImageInfo(CL_IMAGE_WIDTH, &cl_image2d_width);
          cl_image->getImageInfo(CL_IMAGE_HEIGHT, &cl_image2d_height);

          const bool fp16_support =
              CLRuntime::Global()->get_precision() ==
       lite_api::CL_PRECISION_FP16;
          if (fp16_support) {
            auto* h_image = static_cast<half_t*>(TargetWrapperCL::MapImage(
                d_image, cl_image2d_width, cl_image2d_height, 0, 0));
            axis_ = Half2Float(h_image[0]);
            TargetWrapperCL::Unmap(d_image, h_image);
          } else {
            auto* h_image = static_cast<float*>(TargetWrapperCL::MapImage(
                d_image, cl_image2d_width, cl_image2d_height, 0, 0));
            axis_ = h_image[0];
            TargetWrapperCL::Unmap(d_image, h_image);
          }
        }
    */
    axis_ = axis_ >= 0 ? axis_ : axis_ + concat_param_->x[0]->dims().size();

    auto inputs = concat_param_->x;
    auto output_tensor_dims = concat_param_->output->dims();
    auto new_dims = Broadcast2GpuShape(output_tensor_dims);
    flag_ = 4 - output_tensor_dims.size() + axis_;

    for (auto& input : inputs) {
      cl_int4 in_shape = {1, 1, 1, 1};
      for (auto j = 0; j < input->dims().size(); ++j) {
        in_shape.s[4 - input->dims().size() + j] = input->dims()[j];
      }
      if (in_shape.s[1] % 4 != 0) {
        align_ = false;
      }
    }

    if (!(flag_ == 1 && !align_) && inputs.size() <= 6) {
      kernel_func_name_ = "Concat" + std::to_string(inputs.size()) +
                          "InputAxis" + std::to_string(flag_);
    } else if (inputs.size() == 2 && flag_ == 1) {
      kernel_func_name_ = "Concat2InputAxis1Common";
    } else if (inputs.size() == 3 && axis_ == 1 &&
               output_tensor_dims.size() == 4) {
      kernel_func_name_ = "concatByCWith3Inputs";
    } else if (inputs.size() == 4 && axis_ == 1 &&
               output_tensor_dims.size() == 4) {
      kernel_func_name_ = "concatByCWith4Inputs";
    } else {
      // note: do layout transform between image and buffer,
      // before and after concat(buffer impl.)
      kernel_func_name_ = "concat_mul_buffer";  // buffer/concat_kernel.cl
      build_options_ = " -DCL_DTYPE_float";
      auto in_dims = inputs[0]->dims();
      for (int i = 0; i < axis_; i++) {
        pre_size_ *= in_dims[i];
      }
      for (int i = axis_ + 1; i < in_dims.size(); i++) {
        post_size_ *= in_dims[i];
      }
      // create kernels of img2buf and buf2img
      auto img_to_buf_kernels = KernelRegistry::Global().Create(
          "layout", TARGET(kOpenCL), PRECISION(kAny), DATALAYOUT(kNCHW));
      auto buf_to_img_kernels =
          KernelRegistry::Global().Create("layout",
                                          TARGET(kOpenCL),
                                          PRECISION(kAny),
                                          DATALAYOUT(kImageDefault));

      img_to_buf_kernel_ = std::move(img_to_buf_kernels.front());
      buf_to_img_kernel_ = std::move(buf_to_img_kernels.front());
    }

    auto input0_tensor_dims = inputs[0]->dims();
    for (int i = 1; i < inputs.size(); i++) {
      auto dims = inputs[i]->dims();
      CHECK(input0_tensor_dims.size() == dims.size())
          << "All inputs must have the same axes!";
      for (int i = 0; i < dims.size(); i++) {
        if (i != axis_) {
          CHECK(input0_tensor_dims[i] == dims[i])
              << "All inputs must have the same shape, except at concat axis!";
        }
      }
    }

    std::string kernel_file;
    if (kernel_func_name_ == "concat_mul_buffer") {
      kernel_file = "buffer/concat_kernel.cl";
    } else if (kernel_func_name_ == "Concat2InputAxis1" ||
               kernel_func_name_ == "Concat2InputAxis1Common") {
      kernel_file = "image/concat_default_kernel.cl";
    } else {
      kernel_file = "image/concat_kernel.cl";
    }

    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;

    context.cl_context()->AddKernel(
        kernel_func_name_, kernel_file, build_options_, time_stamp_);
  }

  void Run() override {
    const auto& output_tensor_dims = concat_param_->output->dims();
    auto new_dim = Broadcast2GpuShape(output_tensor_dims);
    int output_tensor_w = new_dim[3];
    int output_tensor_c = new_dim[1];
    auto output_image_shape = InitImageDimInfoWith(output_tensor_dims);
    auto* output_image_p = MUTABLE_DATA_GPU(concat_param_->output,
                                            output_image_shape["width"],
                                            output_image_shape["height"],
                                            nullptr);
    auto inputs = concat_param_->x;

    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(
                        output_tensor_dims[output_tensor_dims.size() - 1]),
                    static_cast<cl::size_type>(
                        output_image_shape["width"] /
                        output_tensor_dims[output_tensor_dims.size() - 1]),
                    static_cast<cl::size_type>(output_image_shape["height"])};
    if (!(flag_ == 1 && !align_) && inputs.size() <= 6) {
      global_work_size =
          cl::NDRange{static_cast<cl::size_type>(
                          output_image_shape["width"] /
                          output_tensor_dims[output_tensor_dims.size() - 1]),
                      static_cast<cl::size_type>(
                          output_tensor_dims[output_tensor_dims.size() - 1]),
                      static_cast<cl::size_type>(output_image_shape["height"])};
    }

#ifdef LITE_WITH_LOG
    VLOG(4) << "concat input shape:";
    for (size_t i = 0; i < inputs.size(); i++) {
      VLOG(4) << "inputs [" << i << "]"
              << "   dims: " << inputs[i]->dims();
    }

    VLOG(4) << "concat output shape:";
    VLOG(4) << " out dims: " << output_tensor_dims;
    VLOG(4) << "axis_: " << axis_;
    VLOG(4) << "flag_: " << flag_;
    VLOG(4) << "align_: " << ((align_ == true) ? "true" : "false");

    VLOG(4) << TargetToStr(concat_param_->output->target());
    VLOG(4) << "output_image_shape(w,h): " << output_image_shape["width"] << " "
            << output_image_shape["height"];
    VLOG(4) << "output_tensor_w: " << output_tensor_w;
    VLOG(4) << "global_work_size: " << global_work_size[0] << "  "
            << global_work_size[1] << "  " << global_work_size[2];
    VLOG(4) << "kernel_func_name: " << kernel_func_name_;
#endif

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());

    if (!(flag_ == 1 && !align_) && inputs.size() <= 6) {
      cl_int status;
      int arg_cnt = 0;
      for (auto& input : inputs) {
        auto* input_image_p = GET_DATA_GPU(input);
        status = kernel.setArg(arg_cnt++, *input_image_p);
        CL_CHECK_FATAL(status);
      }
      status = kernel.setArg(arg_cnt++, *output_image_p);
      CL_CHECK_FATAL(status);

      for (auto& input : inputs) {
        cl_int4 in_shape = {1, 1, 1, 1};
        for (auto j = 0; j < input->dims().size(); ++j) {
          in_shape.s[4 - input->dims().size() + j] = input->dims()[j];
        }
        in_shape.s[1] = UP_DIV(in_shape.s[1], 4);
        status = kernel.setArg(arg_cnt++, in_shape);
        CL_CHECK_FATAL(status);
      }
      cl_int4 out_shape = {1, 1, 1, 1};
      for (auto j = 0; j < output_tensor_dims.size(); ++j) {
        out_shape.s[4 - output_tensor_dims.size() + j] = output_tensor_dims[j];
      }
      out_shape.s[1] = UP_DIV(out_shape.s[1], 4);
      status = kernel.setArg(arg_cnt++, out_shape);
      CL_CHECK_FATAL(status);

      status = EnqueueNDRangeKernel(context,
                                    kernel,
                                    cl::NullRange,
                                    global_work_size,
                                    cl::NullRange,
                                    nullptr,
                                    event_);
      CL_CHECK_FATAL(status);
    } else if (kernel_func_name_ == "Concat2InputAxis1Common") {
      auto* input0_image_p = GET_DATA_GPU(inputs[0]);
      auto* input1_image_p = GET_DATA_GPU(inputs[1]);
      int input0_dims_axis = inputs[0]->dims()[axis_];
      cl_int status = kernel.setArg(0, *input0_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(1, *input1_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(2, *output_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, input0_dims_axis);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(4, output_tensor_w);
      CL_CHECK_FATAL(status);

      status = EnqueueNDRangeKernel(context,
                                    kernel,
                                    cl::NullRange,
                                    global_work_size,
                                    cl::NullRange,
                                    nullptr,
                                    event_);
      CL_CHECK_FATAL(status);
    } else if (kernel_func_name_ == "concatByCWith3Inputs" ||
               kernel_func_name_ == "concatByCWith4Inputs") {
      auto* input0 = inputs[0];
      auto* input0_image_p = GET_DATA_GPU(input0);
      int input0_tensor_c = input0->dims()[1];

      auto* input1 = inputs.size() >= 2 ? inputs[1] : nullptr;
      auto* input1_image_p = input1 ? GET_DATA_GPU(input1) : nullptr;
      int input1_tensor_c = input1 ? input1->dims()[1] : -1;

      auto* input2 = inputs.size() >= 3 ? inputs[2] : nullptr;
      auto* input2_image_p = input2 ? GET_DATA_GPU(input2) : nullptr;
      int input2_tensor_c = input2 ? input2->dims()[1] : -1;

      auto* input3 = inputs.size() >= 4 ? inputs[3] : nullptr;
      auto* input3_image_p = input3 ? GET_DATA_GPU(input3) : nullptr;
      int input3_tensor_c = input3 ? input3->dims()[1] : -1;

      int output_tensor_c = output_tensor_dims[1];
      int output_tensor_w = output_tensor_dims[3];

      const std::vector<size_t>& default_work_size = DefaultGlobalWorkSize(
          output_tensor_dims,
          DDim(std::vector<DDim::value_type>{
              static_cast<int64_t>(output_image_shape["width"]),
              static_cast<int64_t>(output_image_shape["height"])}));
      cl::NDRange global_work_size =
          cl::NDRange{static_cast<size_t>(default_work_size[0]),
                      static_cast<size_t>(default_work_size[1]),
                      static_cast<size_t>(default_work_size[2])};

      cl_int status;
      status = kernel.setArg(0, *output_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(1, output_tensor_c);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(2, output_tensor_w);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(3, *input0_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(4, input0_tensor_c);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(5, *input1_image_p);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(6, input1_tensor_c);
      CL_CHECK_FATAL(status);
      if (inputs.size() >= 3) {
        status = kernel.setArg(7, *input2_image_p);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(8, input2_tensor_c);
        CL_CHECK_FATAL(status);
      }
      if (inputs.size() == 4) {
        status = kernel.setArg(9, *input3_image_p);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(10, input3_tensor_c);
        CL_CHECK_FATAL(status);
      }
      status = EnqueueNDRangeKernel(context,
                                    kernel,
                                    cl::NullRange,
                                    global_work_size,
                                    cl::NullRange,
                                    nullptr,
                                    event_);
      CL_CHECK_FATAL(status);
    } else if (kernel_func_name_ == "concat_mul_buffer") {
      // note: do image layout transform: image to buffer
      size_t inputs_num = inputs.size();
      std::vector<const cl::Image2D*> inputs_image_pointers(inputs_num);
      std::vector<std::map<std::string, size_t>> inputs_image_shapes(
          inputs_num);
      std::vector<DDimLite> inputs_dims(inputs_num);
      std::vector<cl::Buffer*> inputs_buffer_pointers(inputs_num);
      for (int i = 0; i < inputs_num; i++) {
        auto* input = inputs[i];
        inputs_dims[i] = input->dims();
        inputs_image_shapes[i] = InitImageDimInfoWith(input->dims());
        inputs_image_pointers[i] = GET_DATA_GPU(input);
      }
      // create and set param, context to kernel img_to_buf
      std::vector<operators::LayoutParam> img_to_buf_params(inputs_num);
      std::vector<lite::Tensor> outputs_vec(inputs_num);
      std::vector<cl::Buffer*> outputs_buffer_pointers(inputs_num);
      for (size_t i = 0; i < inputs_num; ++i) {
        img_to_buf_params[i].x = inputs[i];
        img_to_buf_params[i].y = &outputs_vec[i];
        outputs_vec[i].Resize(inputs_dims[i]);
        outputs_buffer_pointers[i] =
            outputs_vec[i].mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
        img_to_buf_kernel_->SetParam(img_to_buf_params[i]);

        std::unique_ptr<KernelContext> img_to_buf_context(new KernelContext);
        context.CopySharedTo(&(img_to_buf_context->As<OpenCLContext>()));
        img_to_buf_kernel_->SetContext(std::move(img_to_buf_context));
        img_to_buf_kernel_->Launch();
      }
      // create and set param, context to kernel buf_to_img
      std::shared_ptr<lite::Tensor> concat_mul_buf_output_t(new lite::Tensor);
      concat_mul_buf_output_t->Resize(concat_param_->output->dims());
      auto conat_mul_buf_output_data =
          concat_mul_buf_output_t->mutable_data<float, cl::Buffer>(
              TARGET(kOpenCL));
      operators::LayoutParam buf_to_img_param;
      buf_to_img_param.x = concat_mul_buf_output_t.get();
      buf_to_img_param.y = concat_param_->output;
      buf_to_img_kernel_->SetParam(buf_to_img_param);

      std::unique_ptr<KernelContext> buf_to_img_context(new KernelContext);
      context.CopySharedTo(&(buf_to_img_context->As<OpenCLContext>()));
      buf_to_img_kernel_->SetContext(std::move(buf_to_img_context));

      int cur_axis_start_idx = 0;
      int total = output_tensor_dims[axis_] * post_size_;
      for (size_t i = 0; i < inputs_num; ++i) {
        auto* x_buf = outputs_buffer_pointers[i];
        int axis_dim_size = inputs[i]->dims()[axis_];
        global_work_size = cl::NDRange{static_cast<size_t>(post_size_),
                                       static_cast<size_t>(axis_dim_size),
                                       static_cast<size_t>(pre_size_)};
        int total0 = axis_dim_size * post_size_;
#ifdef LITE_WITH_LOG
        VLOG(2) << "--------------- i:" << i << " -----------------";
        VLOG(2) << "post_size_:" << post_size_;
        VLOG(2) << "pre_size_:" << pre_size_;
        VLOG(2) << "axis_dim_size:" << axis_dim_size;
        VLOG(2) << "cur_axis_start_idx:" << cur_axis_start_idx;
        VLOG(2) << "total:" << total;
        VLOG(2) << "total0:" << total0;
#endif
        cl_int status;
        status = kernel.setArg(0, *x_buf);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(1, *conat_mul_buf_output_data);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(2, cur_axis_start_idx);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(3, total);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(4, total0);
        CL_CHECK_FATAL(status);

        status = EnqueueNDRangeKernel(context,
                                      kernel,
                                      cl::NullRange,
                                      global_work_size,
                                      cl::NullRange,
                                      nullptr,
                                      event_);
        CL_CHECK_FATAL(status);
        cur_axis_start_idx += axis_dim_size;
      }
      // run kernel: buffer->image
      buf_to_img_kernel_->Launch();
    }
  }

  std::string doc() { return "Concat using cl::Image, kFP16"; }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  int axis_ = 1;
  int flag_ = 1;
  int pre_size_ = 1;
  int post_size_ = 1;
  int align_{true};
  param_t* concat_param_{nullptr};
  std::string kernel_func_name_{};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  std::unique_ptr<KernelBase> img_to_buf_kernel_;
  std::unique_ptr<KernelBase> buf_to_img_kernel_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::opencl::ConcatComputeImage Concat_image;

REGISTER_LITE_KERNEL(
    concat, kOpenCL, kFP16, kImageDefault, Concat_image, ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindInput("AxisTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .Finalize();
