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

#include "lite/kernels/opencl/slice_buffer_compute.h"
#include <algorithm>
#include <string>
#include <vector>
#if defined(_MSC_VER)
#undef min
#undef max
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

inline static std::vector<int32_t> get_new_data_from_tensorlist(
    const std::vector<lite::Tensor*>& list_new_data_tensor) {
  // get tensor
  std::vector<int32_t> vec_new_data;
  for (size_t i = 0; i < list_new_data_tensor.size(); ++i) {
    auto tensor = list_new_data_tensor[i];
    CHECK_EQ(tensor->dims(), DDim({1})) << "shape of dim tensor should be [1]";
    vec_new_data.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
  }
  return vec_new_data;
}

inline static std::vector<int32_t> get_new_data_from_tensor(
    const lite::Tensor* new_data_tensor) {
  std::vector<int32_t> vec_new_data;
  auto* new_data = new_data_tensor->data<int32_t>();
  vec_new_data =
      std::vector<int32_t>(new_data, new_data + new_data_tensor->numel());
  return vec_new_data;
}

template <typename T, PrecisionType PType>
void SliceCompute<T, PType>::PrepareForRun() {
  auto& context = this->ctx_->template As<OpenCLContext>();
  slice_param_ = this->param_.template get_mutable<param_t>();
  auto param = *slice_param_;

  if (std::is_same<T, float>::value) {
    build_options_ += " -DDTYPE=" + std::string{"float"};
  } else if (std::is_same<T, int>::value) {
    build_options_ += " -DDTYPE=" + std::string{"int"};
  } else if (std::is_same<T, int64_t>::value) {
    build_options_ += " -DDTYPE=" + std::string{"long"};
  } else {
    LOG(FATAL) << "Not support data type";
  }

  VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
  context.cl_context()->AddKernel(
      kernel_func_name_, "buffer/slice_kernel.cl", build_options_, time_stamp_);

  auto in = param.X;
  auto in_dims = in->dims();
  auto out = param.Out;
  auto out_dims = out->dims();

  std::vector<int> axes = param.axes;
  std::vector<int32_t> starts = param.starts;
  std::vector<int32_t> ends = param.ends;
  std::vector<int> decrease_axis = param.decrease_axis;
  std::vector<int> infer_flags = param.infer_flags;

  auto list_new_ends_tensor = param.EndsTensorList;
  auto list_new_starts_tensor = param.StartsTensorList;

  bool need_infer = false;
  if (param.StartsTensor || param.EndsTensor) {
    need_infer = true;
  }
  if (list_new_starts_tensor.size() > 0 || list_new_ends_tensor.size() > 0) {
    need_infer = true;
  }
  if (need_infer) {
    if (param.StartsTensor) {
      starts = get_new_data_from_tensor(param.StartsTensor);
    } else if (list_new_starts_tensor.size() > 0) {
      starts = get_new_data_from_tensorlist(list_new_starts_tensor);
    }
    CHECK_EQ(starts.size(), axes.size())
        << "The size of starts must be equal to the size of axes.";
    if (param.EndsTensor) {
      ends = get_new_data_from_tensor(param.EndsTensor);
    } else if (list_new_ends_tensor.size() > 0) {
      ends = get_new_data_from_tensorlist(list_new_ends_tensor);
    }
    CHECK_EQ(ends.size(), axes.size())
        << "The size of ends must be equal to the size of axes.";
    out_dims = in_dims;
    int dim_value, start, end;
    for (size_t i = 0; i < axes.size(); ++i) {
      dim_value = out_dims[axes[i]];
      if (dim_value > 0) {
        // when end = start+1 and start == -1
        if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
          auto ret =
              std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
          if (ret != decrease_axis.end()) {
            ends[i] = 10000000;
          }
        }

        start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
        end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
        start = std::max(start, 0);
        end = std::max(end, 0);
        end = std::min(end, dim_value);
        CHECK_GT(end, start) << "end should greater than start";
        out_dims[axes[i]] = end - start;
      }
    }
    out->Resize(out_dims);
    // generate new shape
    if (decrease_axis.size() > 0) {
      std::vector<int64_t> new_out_shape;
      for (size_t i = 0; i < decrease_axis.size(); ++i) {
        CHECK_EQ(out_dims[decrease_axis[i]], 1) << "decrease dim should be 1";
        out_dims[decrease_axis[i]] = 0;
      }

      for (int i = 0; i < out_dims.size(); ++i) {
        if (out_dims[i] != 0) {
          new_out_shape.push_back(out_dims[i]);
        }
      }
      if (new_out_shape.size() == 0) {
        new_out_shape.push_back(1);
      }
      DDim new_dims;
      new_dims.ConstructFrom(new_out_shape);
      out_dims = new_dims;
    }
  }

  // resize out dims
  if (decrease_axis.size() > 0) {
    if (decrease_axis.size() == static_cast<size_t>(in_dims.size())) {
      std::vector<int64_t> vec_origin_out_shape(decrease_axis.size(), 1);
      out->Resize(DDim(vec_origin_out_shape));
    } else {
      std::vector<int64_t> vec_origin_out_shape(
          out_dims.size() + decrease_axis.size(), -1);

      for (size_t i = 0; i < decrease_axis.size(); ++i) {
        vec_origin_out_shape[decrease_axis[i]] = 1;
      }

      int index = 0;
      for (size_t i = 0; i < vec_origin_out_shape.size(); ++i) {
        if (vec_origin_out_shape[i] == -1) {
          vec_origin_out_shape[i] = out_dims[index];
          ++index;
        }
      }

      out->Resize(DDim(vec_origin_out_shape));
    }
  }
}

template <typename T, PrecisionType PType>
void SliceCompute<T, PType>::ReInitWhenNeeded() {
  const auto in_dims = slice_param_->X->dims();
  if ((!first_epoch_for_reinit_ && in_dims != last_x_dims_) ||
      first_epoch_for_reinit_) {
    last_x_dims_ = in_dims;
    first_epoch_for_reinit_ = false;

    // calculate tmp variables
    const int LEN = in_dims.size();
    const int MEM_SIZE = LEN * sizeof(int32_t);
    std::vector<int> axes = slice_param_->axes;
    std::vector<int32_t> starts = slice_param_->starts;
    std::vector<int32_t> ends = slice_param_->ends;
    auto out_dims = in_dims;
    std::vector<int> real_starts(in_dims.size(), 0);
    for (int i = 0; i < axes.size(); i++) {
      int dim_value = in_dims[axes[i]];
      if (dim_value > 0) {
        int start = starts[i] < 0 ? (starts[i] + dim_value) : starts[i];
        int end = ends[i] < 0 ? (ends[i] + dim_value) : ends[i];
        start = std::max(start, 0);
        end = std::max(end, 0);
        end = std::min(end, dim_value);
        out_dims[axes[i]] = end - start;
        real_starts[axes[i]] = start;
      }
    }
    std::vector<int> dst_step(LEN);
    for (int i = 0; i < in_dims.size(); ++i) {
      dst_step[i] = 1;
    }
    std::vector<int> src_step(LEN);
    for (int i = 0; i < in_dims.size(); ++i) {
      src_step[i] = 1;
    }
    out_num_ = out_dims[in_dims.size() - 1];
    for (int i = in_dims.size() - 2; i >= 0; i--) {
      dst_step[i] = out_dims[i + 1] * dst_step[i + 1];
      src_step[i] = in_dims[i + 1] * src_step[i + 1];
      out_num_ *= out_dims[i];
    }

    // malloc temporary GPU data
    src_step_buf_ = static_cast<cl::Buffer*>(TargetWrapperCL::Malloc(MEM_SIZE));
    dst_step_buf_ = static_cast<cl::Buffer*>(TargetWrapperCL::Malloc(MEM_SIZE));
    real_starts_buf_ =
        static_cast<cl::Buffer*>(TargetWrapperCL::Malloc(MEM_SIZE));
    TargetWrapperCL::MemcpySync(
        src_step_buf_, src_step.data(), MEM_SIZE, IoDirection::HtoD);
    TargetWrapperCL::MemcpySync(
        dst_step_buf_, dst_step.data(), MEM_SIZE, IoDirection::HtoD);
    TargetWrapperCL::MemcpySync(
        real_starts_buf_, real_starts.data(), MEM_SIZE, IoDirection::HtoD);

    auto& context = this->ctx_->template As<OpenCLContext>();
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());

    gws_ = cl::NDRange{static_cast<size_t>(out_num_)};
  }
}

template <typename T, PrecisionType PType>
void SliceCompute<T, PType>::Run() {
  auto& context = this->ctx_->template As<OpenCLContext>();
  CHECK(context.cl_context() != nullptr);
  auto* x_buf = slice_param_->X->template data<T, cl::Buffer>();
  auto* out_buf =
      slice_param_->Out->template mutable_data<T, cl::Buffer>(TARGET(kOpenCL));

  auto in = slice_param_->X;
  std::vector<int> in_dims(in->dims().size());
  for (auto i = 0; i < in->dims().size(); i++) {
    in_dims[i] = in->dims()[i];
  }

  cl_int status;
  int arg_idx = 0;
  auto kernel = kernel_;
  CHECK(src_step_buf_ != nullptr);
  status = kernel.setArg(arg_idx++, *x_buf);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(arg_idx++, *out_buf);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(arg_idx++, *src_step_buf_);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(arg_idx++, *dst_step_buf_);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(arg_idx++, *real_starts_buf_);
  CL_CHECK_FATAL(status);
  status = kernel.setArg(arg_idx++, static_cast<int>(in_dims.size()));
  CL_CHECK_FATAL(status);
  status = kernel.setArg(arg_idx++, out_num_);
  CL_CHECK_FATAL(status);

  status = EnqueueNDRangeKernel(context,
                                kernel,
                                cl::NullRange,
                                gws_,
                                cl::NullRange,
                                nullptr,
                                this->event_);
  CL_CHECK_FATAL(status);
}

#ifdef LITE_WITH_PROFILE
void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
  ch->kernel_func_name = kernel_func_name_;
  ch->cl_event =
      this->event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
}
#endif

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using slice_float =
    paddle::lite::kernels::opencl::SliceCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(slice, kOpenCL, kFloat, kNCHW, slice_float, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kFloat))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kFloat))})
    .Finalize();

using slice_int32 =
    paddle::lite::kernels::opencl::SliceCompute<int, PRECISION(kInt32)>;
REGISTER_LITE_KERNEL(slice, kOpenCL, kInt32, kNCHW, slice_int32, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kInt32))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kInt32))})
    .Finalize();

using slice_int64 =
    paddle::lite::kernels::opencl::SliceCompute<int64_t, PRECISION(kInt64)>;
REGISTER_LITE_KERNEL(slice, kOpenCL, kInt64, kNCHW, slice_int64, def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kInt64))})
    .BindInput("StartsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("StartsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("EndsTensorList",
               {LiteType::GetTensorListTy(TARGET(kARM), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL), PRECISION(kInt64))})
    .Finalize();
