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

/*
 * This file implements BasicProfile, a profiler that helps to profile the basic
 * CPU execution. It can display the min, max, average lantency of the execution
 * of each kernel.
 */
#pragma once
#include <string>
#include <vector>
#include "lite/core/program.h"

#ifdef LITE_WITH_OPENCL
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/kernels/opencl/image_helper.h"
#endif

namespace paddle {
namespace lite {
namespace profile {

template <typename dtype>
static void write_tensorfile(const Tensor* tensor, const std::string& locate) {
  if (locate.find('/') != std::string::npos) {
    return;
  }
  FILE* fp = fopen(locate.c_str(), "w");
  if (fp == nullptr) {
    LOG(ERROR) << "file open field " << locate;
  } else {
    const dtype* data = tensor->data<dtype>();
    for (int i = 0; i < tensor->numel(); ++i) {
      fprintf(fp, "[%d] %f \n", i, static_cast<float>(data[i]));
    }
  }
  fclose(fp);
}

class PrecisionProfiler {
 public:
  // TODO(ysh329): need to remove `explicit PrecisionProfiler`
  // keep this method only for arm/math/conditional
  explicit PrecisionProfiler(const Instruction* inst) {
    std::string inst_precison_str = GetInstPrecision(inst);
  }

  PrecisionProfiler() {}

  std::string GetSummaryHeader() {
    using std::setw;
    using std::left;
    using std::fixed;
    STL::stringstream ss;
    ss << "========================================= "
       << "Detailed Precision Profiler Summary "
       << "=========================================" << std::endl;
    ss << setw(45) << left << "operator:(kernel_info)"
       << " " << setw(70) << left << "output_tensor_name:(tensor_info)"
       << " " << setw(15) << left << "tensor_dims"
       << " " << setw(15) << left << "tensor_mean"
       << " " << setw(15) << left << "tensor_standard_deviation" << std::endl;

    return ss.str();
  }

  template <typename T>
  double compute_mean(const T* in, const size_t length) {
    double sum = 0.;
    for (size_t i = 0; i < length; ++i) {
      sum += in[i];
    }
    return sum / length;
  }

  template <typename T>
  double compute_standard_deviation(const T* in,
                                    const size_t length,
                                    bool has_mean = false,
                                    double mean = 10000) {
    if (!has_mean) {
      mean = compute_mean<T>(in, length);
    }

    double variance = 0.;
    for (size_t i = 0; i < length; ++i) {
      variance += pow((in[i] - mean), 2);
    }
    variance /= length;
    return sqrt(variance);
  }

  // check if output tensor unused
  bool is_unused(const Tensor* in) {
    if (!in->data<int8_t>()) {
      return true;
    }
    return false;
  }

  void compute_tensor_precision_info(const Tensor* in,
                                     TargetType target_type,
                                     PrecisionType precision_type,
                                     DataLayoutType layout_type,
                                     double* mean,
                                     double* std_dev,
                                     std::string name = "inst") {
    std::string unsupported_error_log =
        "Unsupported precision profile for kernel registered on" +
        TargetToStr(target_type) + "/" + PrecisionToStr(precision_type) + "/" +
        DataLayoutToStr(layout_type);

    if (target_type == TARGET(kARM) || target_type == TARGET(kHost) ||
        target_type == TARGET(kX86)) {
      switch (precision_type) {
        case PRECISION(kFloat): {
          auto ptr = in->data<float>();
          // write_tensorfile<float>(in, name);
          *mean = compute_mean<float>(ptr, in->numel());
          *std_dev =
              compute_standard_deviation<float>(ptr, in->numel(), true, *mean);
          return;
        }
        case PRECISION(kAny): {
          auto ptr = in->data<float>();
          // write_tensorfile<float>(in, name);
          *mean = compute_mean<float>(ptr, in->numel());
          *std_dev =
              compute_standard_deviation<float>(ptr, in->numel(), true, *mean);
          return;
        }
        case PRECISION(kInt8): {
          auto ptr = in->data<int8_t>();
          // write_tensorfile<int8_t>(in, name);
          *mean = compute_mean<int8_t>(ptr, in->numel());
          *std_dev =
              compute_standard_deviation<int8_t>(ptr, in->numel(), true, *mean);
          return;
        }
        case PRECISION(kInt32): {
          auto ptr = in->data<int32_t>();
          // write_tensorfile<int32_t>(in, name);
          *mean = compute_mean<int32_t>(ptr, in->numel());
          *std_dev = compute_standard_deviation<int32_t>(
              ptr, in->numel(), true, *mean);
          return;
        }
        default:
          *mean = -333333333333;
          *std_dev = -33333333333;
          LOG(ERROR) << unsupported_error_log;
          return;
      }
#ifdef LITE_WITH_OPENCL
    } else if (target_type == TARGET(kOpenCL)) {
      switch (layout_type) {
        case DATALAYOUT(kImageDefault): {
          paddle::lite::CLImageConverterDefault default_convertor;
          auto image_shape = default_convertor.InitImageDimInfoWith(in->dims());
          size_t im_w = image_shape[0];
          size_t im_h = image_shape[1];
          VLOG(1) << "image shape(W,H) of " << name << ": " << im_w << " "
                  << im_h;
          std::vector<uint16_t> in_data_v(im_w * im_h * 4);
          std::vector<float> real_out_v(in->numel());
          const size_t cl_image2d_row_pitch{0};
          const size_t cl_image2d_slice_pitch{0};
          TargetWrapperCL::ImgcpySync(in_data_v.data(),
                                      in->data<uint16_t, cl::Image2D>(),
                                      im_w,
                                      im_h,
                                      cl_image2d_row_pitch,
                                      cl_image2d_slice_pitch,
                                      IoDirection::DtoH);
          default_convertor.ImageToNCHW(
              in_data_v.data(), real_out_v.data(), image_shape, in->dims());
          // write_tensorfile<float>(in, name);
          CHECK(real_out_v.size() == in->numel());
          *mean = compute_mean<float>(real_out_v.data(), real_out_v.size());
          *std_dev = compute_standard_deviation<float>(
              real_out_v.data(), in->numel(), true, *mean);
          return;
        }
        case DATALAYOUT(kNCHW): {
          std::vector<float> in_data_v(in->numel(), 0);
          TargetWrapperCL::MemcpySync(in_data_v.data(),
                                      in->data<float>(),
                                      in->numel() * sizeof(float),
                                      IoDirection::DtoH);
          VLOG(1) << name << ":" << in->numel();
          *mean = compute_mean<float>(in_data_v.data(), in->numel());
          *std_dev = compute_standard_deviation<float>(
              in_data_v.data(), in->numel(), true, *mean);
          return;
        }
        default:
          *mean = -222222222222;
          *std_dev = -22222222222;
          LOG(ERROR) << unsupported_error_log;
          return;
      }
#endif
    } else {
      *mean = -111111111111;
      *std_dev = -11111111111;
      LOG(ERROR) << unsupported_error_log;
      return;
    }
  }

  std::string GetInstPrecision(const Instruction* inst = nullptr) {
    using std::setw;
    using std::left;
    using std::fixed;
    STL::stringstream ss;

    VLOG(1) << ">> Running kernel: " << inst->op()->op_info()->Repr()
            << " registered on " << TargetToStr(inst->kernel()->target()) << "/"
            << PrecisionToStr(inst->kernel()->precision()) << "/"
            << DataLayoutToStr(inst->kernel()->layout());

    std::string kernel_repr = inst->op()->op_info()->Repr();
    std::string kernel_place = TargetToStr(inst->kernel()->target()) + "/" +
                               PrecisionToStr(inst->kernel()->precision()) +
                               "/" + DataLayoutToStr(inst->kernel()->layout());
    std::string op_name = inst->op()->op_info()->Type();

    if (inst->op()->op_info()->Type() != "fetch") {
      auto op = const_cast<lite::OpLite*>(inst->op());
      auto kernel = inst->kernel();
      auto op_scope = op->scope();
      auto out_names = op->op_info()->output_names();
      for (auto& out_name : out_names) {
        std::string out_arg_name;
        op->op_info()->GetOutputArgname(out_name, &out_arg_name);
        auto type = kernel->GetOutputDeclType(out_arg_name);

        if (type->IsTensor()) {
          const Tensor* tout =
              op_scope->FindVar(out_name)->GetMutable<Tensor>();
          double mean = -999999;
          double std_dev = -100000;
          std::string mean_str{"unused"};
          std::string std_dev_str{"unused"};

          if (!is_unused(tout)) {
            compute_tensor_precision_info(tout,
                                          type->target(),
                                          type->precision(),
                                          type->layout(),
                                          &mean,
                                          &std_dev,
                                          out_name);
            mean_str = std::to_string(mean);
            std_dev_str = std::to_string(std_dev);
          }
          std::string kernel_info = op_name + ":" + kernel_place;
          std::string output_arg_info = out_name + ":" +
                                        TargetToStr(type->target()) + "/" +
                                        PrecisionToStr(type->precision()) +
                                        "/" + DataLayoutToStr(type->layout());

          ss << setw(45) << left << kernel_info << " " << setw(70) << left
             << output_arg_info << " " << setw(15) << left << tout->dims()
             << " " << setw(15) << left << mean_str << " " << setw(15) << left
             << std_dev_str << std::endl;
        } else if (type->IsTensorList()) {
          auto touts =
              op_scope->FindVar(out_name)->GetMutable<std::vector<Tensor>>();
          for (auto t : *touts) {
            const Tensor* tout = &t;
            double mean = -999999;
            double std_dev = -100000;
            std::string mean_str{"unused"};
            std::string std_dev_str{"unused"};

            if (!is_unused(tout)) {
              compute_tensor_precision_info(tout,
                                            type->target(),
                                            type->precision(),
                                            type->layout(),
                                            &mean,
                                            &std_dev,
                                            out_name);
              mean_str = std::to_string(mean);
              std_dev_str = std::to_string(std_dev);
            }
            std::string kernel_info = op_name + ":" + kernel_place;
            std::string output_arg_info = out_name + ":" +
                                          TargetToStr(type->target()) + "/" +
                                          PrecisionToStr(type->precision()) +
                                          "/" + DataLayoutToStr(type->layout());

            ss << setw(45) << left << kernel_info << " " << setw(70) << left
               << output_arg_info << " " << setw(15) << left << tout->dims()
               << " " << setw(15) << left << mean_str << " " << setw(15) << left
               << std_dev_str << std::endl;
          }
        }
      }
    }
    return ss.str();
  }
};

}  // namespace profile
}  // namespace lite
}  // namespace paddle

// TODO(ysh329): need to remove.
// keep this method only for arm/math/conditional_block_compute
#define LITE_PRECISION_PROFILE(inst) \
  { auto a = paddle::lite::profile::PrecisionProfiler(&inst); }
