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

#include <time.h>

#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/core/program.h"
#include "lite/utils/io.h"
#ifdef LITE_WITH_X86
#include "lite/backends/x86/fluid/float16.h"
#endif

#ifdef LITE_WITH_OPENCL
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/kernels/opencl/image_helper.h"
#endif

#ifdef LITE_WITH_CUDA
#include "lite/backends/cuda/math/type_trans.h"
#endif

#if defined(_MSC_VER)
#include "lite/backends/x86/port.h"
#endif

namespace paddle {
namespace lite {
namespace profile {

static const std::string get_date_str() {
  std::time_t now = std::time(nullptr);
  char buffer[32];
  if (std::strftime(
          buffer, sizeof(buffer), "%Y-%m-%d_%H-%M-%S", std::localtime(&now))) {
    return std::string(buffer);
  } else {
    LOG(WARNING) << "Convert calendar time error! Use the default timestamp.";
    return "timestamp";
  }
}

inline std::string generate_valid_tensor_name(const std::string& name) {
  std::string new_name("");
  for (size_t i = 0; i < name.length(); ++i) {
    if (name[i] != '/') {
      new_name += name[i];
    } else {
      new_name += "_";
    }
  }
  return new_name;
}

template <typename dtype>
static bool write_tensorfile(const Tensor* tensor,
                             const std::string& tensor_name,
                             const std::string prefix_path) {
  std::string new_tensor_name = generate_valid_tensor_name(tensor_name);
  if (tensor_name.find('/') != std::string::npos) {
    LOG(ERROR) << "--> tensor name is abnormal with '\\':" << tensor_name
               << " !!!, replace with '_'," << new_tensor_name
               << new_tensor_name;
  }

  std::string tensor_save_path = prefix_path + new_tensor_name + ".txt";
  FILE* fp = fopen(tensor_save_path.c_str(), "w");
  if (fp == nullptr) {
    LOG(ERROR) << "failed open file " << tensor_save_path;
    return false;
  } else {
    const dtype* data = tensor->data<dtype>();
    for (int i = 0; i < tensor->numel(); ++i) {
      fprintf(fp, "[%d] %f \n", i, static_cast<float>(data[i]));
    }
  }
  fclose(fp);
  LOG(INFO) << "write tensor " << tensor_name
            << " to file:" << tensor_save_path;
  return true;
}

static bool write_precision_summary_tofile(
    const std::string& string, const std::string& summary_log_dir = "") {
  if (summary_log_dir == "") {
    LOG(INFO) << "The `summary_log_dir` of precision summary file is not set. "
                 "summary_log_dir:"
              << summary_log_dir;
    return false;
  }

  FILE* fp = fopen(summary_log_dir.c_str(), "a");
  if (fp == nullptr) {
    LOG(INFO) << "Open precision summary file:" << summary_log_dir << "failed.";
    return false;
  } else {
    fprintf(fp, "%s\n", string.c_str());
  }
  fclose(fp);
  return true;
}

class PrecisionProfiler {
 public:
  // TODO(ysh329): need to remove `explicit PrecisionProfiler`
  // keep this method only for arm/math/conditional
  explicit PrecisionProfiler(const Instruction* inst) {
    std::string inst_precison_str = GetInstPrecision(inst);
  }

  PrecisionProfiler() {
    MkDirRecur(log_dir_);
    const char* write_to_file_raw =
        std::getenv("PADDLELITE_PRECISION_WRITE_TO_FILE");
    write_result_to_file_ =
        (write_to_file_raw && atoi(write_to_file_raw) > 0) ? true : false;
  }

  std::string GetSummaryHeader() {
    using std::setw;
    using std::left;
    using std::fixed;
    STL::stringstream ss;
    ss << "\n\n========================================= "
       << "Detailed Precision Profiler Summary "
       << "=========================================" << std::endl;
    ss << setw(45) << left << "operator:(kernel_info)"
       << " " << setw(70) << left << "output_tensor_name:(tensor_info)"
       << " " << setw(15) << left << "dims"
       << " " << setw(15) << left << "mean"
       << " " << setw(15) << left << "std_deviation"
       << " " << setw(15) << left << "ave_grow_rate*" << std::endl;

    // write to file with path: `summary_log_dir`
    if (summary_log_dir_ != "") {
      FILE* fp = fopen(summary_log_dir_.c_str(), "a");
      std::string header_str{ss.str()};
      fprintf(fp, "%s\n", header_str.c_str());
      fclose(fp);
    }
    return ss.str();
  }

  std::string GetSummaryTail() {
    STL::stringstream ss;
    ss << "[note]" << std::endl;
    ss << "1. `ave_grow_rate`: show the sequence value of tensor when std_dev "
          "& mean are same."
       << std::endl;
    ss << "2. Enable write each output tensor to file: `export "
          "PADDLELITE_PRECISION_WRITE_TO_FILE=1` on ADB command line."
       << std::endl;
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

  template <typename T>
  double compute_average_grow_rate(const T* in, const size_t length) {
    const double eps = 1e-5;
    double ave_grow_rate = 0.0f;
    for (size_t i = 1; i < length; ++i) {
      ave_grow_rate += (in[i] - in[i - 1]) / (in[i - 1] + eps);
    }
    ave_grow_rate /= length;
    return ave_grow_rate;
  }

  // check if output tensor unused
  bool is_unused(const Tensor* in) {
    if (!in->data<int8_t>()) {
      return true;
    }
    return false;
  }

  std::string rename_out_for_mem_reuse_pass(const std::string& old_name) {
    if (out_tensor_names_map.find(old_name) == out_tensor_names_map.end()) {
      out_tensor_names_map[old_name] = 1;
    } else {
      ++out_tensor_names_map[old_name];
    }
    std::string new_name =
        old_name + "_" + std::to_string(out_tensor_names_map[old_name]);
    return new_name;
  }

  void compute_tensor_precision_info(const Tensor* in,
                                     const std::string op_name,
                                     DataLayoutType layout_type,
                                     double* mean,
                                     double* std_dev,
                                     double* ave_grow_rate,
                                     std::string name = "inst",
                                     bool write_result_to_file = false) {
    TargetType target_type = in->target();
    PrecisionType precision_type = in->precision();

    std::string unsupported_error_log =
        "Unsupported precision profile for kernel registered on" +
        TargetToStr(target_type) + "/" + PrecisionToStr(precision_type) + "/" +
        DataLayoutToStr(layout_type);

    if (target_type == TARGET(kARM) || target_type == TARGET(kHost) ||
        target_type == TARGET(kX86)) {
      switch (precision_type) {
        case PRECISION(kFloat): {
          auto ptr = in->data<float>();
          *mean = compute_mean<float>(ptr, in->numel());
          *std_dev =
              compute_standard_deviation<float>(ptr, in->numel(), true, *mean);
          *ave_grow_rate = compute_average_grow_rate<float>(ptr, in->numel());
          if (write_result_to_file) {
            write_tensorfile<float>(in, name, log_dir_);
          }
          return;
        }
#ifdef ENABLE_ARM_FP16
        case PRECISION(kFP16): {
          auto ptr = in->data<__fp16>();
          *mean = compute_mean<__fp16>(ptr, in->numel());
          *std_dev =
              compute_standard_deviation<__fp16>(ptr, in->numel(), true, *mean);
          *ave_grow_rate = compute_average_grow_rate<__fp16>(ptr, in->numel());
          if (write_result_to_file) {
            write_tensorfile<__fp16>(in, name, log_dir_);
          }
          return;
        }
#endif
        case PRECISION(kBool): {
          auto ptr = in->data<bool>();
          *mean = compute_mean<bool>(ptr, in->numel());
          *std_dev =
              compute_standard_deviation<bool>(ptr, in->numel(), true, *mean);
          *ave_grow_rate = compute_average_grow_rate<bool>(ptr, in->numel());
          if (write_result_to_file) {
            write_tensorfile<bool>(in, name, log_dir_);
          }
          return;
        }
        case PRECISION(kInt8): {
          auto ptr = in->data<int8_t>();
          *mean = compute_mean<int8_t>(ptr, in->numel());
          *std_dev =
              compute_standard_deviation<int8_t>(ptr, in->numel(), true, *mean);
          *ave_grow_rate = compute_average_grow_rate<int8_t>(ptr, in->numel());
          if (write_result_to_file) {
            write_tensorfile<int8_t>(in, name, log_dir_);
          }
          return;
        }
        case PRECISION(kInt32): {
          auto ptr = in->data<int32_t>();
          *mean = compute_mean<int32_t>(ptr, in->numel());
          *std_dev = compute_standard_deviation<int32_t>(
              ptr, in->numel(), true, *mean);
          *ave_grow_rate = compute_average_grow_rate<int32_t>(ptr, in->numel());
          if (write_result_to_file) {
            write_tensorfile<int32_t>(in, name, log_dir_);
          }
          return;
        }
        case PRECISION(kInt64): {
          auto ptr = in->data<int64_t>();
          *mean = compute_mean<int64_t>(ptr, in->numel());
          *std_dev = compute_standard_deviation<int64_t>(
              ptr, in->numel(), true, *mean);
          if (write_result_to_file) {
            write_tensorfile<int64_t>(in, name, log_dir_);
          }
          return;
        }
        default:
          *mean = -333333333333;
          *std_dev = -33333333333;
          *ave_grow_rate = -33333333333;
          LOG(INFO)
              << "Unsupported precision profile for kernel registered on" +
                     PrecisionToStr(precision_type);
          return;
      }
#ifdef LITE_WITH_OPENCL
    } else if (target_type == TARGET(kOpenCL)) {
      bool use_fp16 = paddle::lite::CLRuntime::Global()->get_precision() ==
                      lite_api::CL_PRECISION_FP16;
      CLRuntime::Global()->command_queue().finish();
      switch (layout_type) {
        case DATALAYOUT(kImageDefault): {
          auto in_dims = in->dims();
          paddle::lite::CLImageConverterDefault default_convertor;
          auto image_shape = default_convertor.InitImageDimInfoWith(in_dims);
          size_t im_w = image_shape[0];
          size_t im_h = image_shape[1];
          VLOG(1) << "image shape(W,H) of " << name << ": " << im_w << " "
                  << im_h;
          auto* in_data_v =
              use_fp16
                  ? static_cast<void*>(
                        calloc(im_w * im_h * 4, sizeof(uint16_t)))
                  : static_cast<void*>(calloc(im_w * im_h * 4, sizeof(float)));

          std::vector<float> real_out_v(in->numel());
          const size_t cl_image2d_row_pitch{0};
          const size_t cl_image2d_slice_pitch{0};
          TargetWrapperCL::ImgcpySync(in_data_v,
                                      use_fp16
                                          ? in->data<uint16_t, cl::Image2D>()
                                          : in->data<float, cl::Image2D>(),
                                      im_w,
                                      im_h,
                                      cl_image2d_row_pitch,
                                      cl_image2d_slice_pitch,
                                      IoDirection::DtoH);
          default_convertor.ImageToNCHW(
              in_data_v, real_out_v.data(), image_shape, in_dims);
          CHECK(real_out_v.size() == in->numel());
          *mean = compute_mean<float>(real_out_v.data(), real_out_v.size());
          *std_dev = compute_standard_deviation<float>(
              real_out_v.data(), in->numel(), true, *mean);
          *ave_grow_rate = compute_average_grow_rate<float>(real_out_v.data(),
                                                            real_out_v.size());
          std::shared_ptr<lite::Tensor> real_out_t(new lite::Tensor);
          real_out_t->Resize(in_dims);
          float* real_out_data = real_out_t->mutable_data<float>();
          memcpy(real_out_data,
                 real_out_v.data(),
                 real_out_v.size() * sizeof(float));
          if (write_result_to_file) {
            write_tensorfile<float>(real_out_t.get(), name, log_dir_);
          }
          return;
        }
        case DATALAYOUT(kImageFolder): {
          auto in_dims = in->dims();
          paddle::lite::CLImageConverterFolder folder_convertor;
          auto image_shape = folder_convertor.InitImageDimInfoWith(in_dims);
          size_t im_w = image_shape[0];
          size_t im_h = image_shape[1];
          VLOG(1) << "image shape(W,H) of " << name << ": " << im_w << " "
                  << im_h;
          auto* in_data_v =
              use_fp16
                  ? static_cast<void*>(
                        calloc(im_w * im_h * 4, sizeof(uint16_t)))
                  : static_cast<void*>(calloc(im_w * im_h * 4, sizeof(float)));

          std::vector<float> real_out_v(in->numel());
          const size_t cl_image2d_row_pitch{0};
          const size_t cl_image2d_slice_pitch{0};
          TargetWrapperCL::ImgcpySync(in_data_v,
                                      use_fp16
                                          ? in->data<uint16_t, cl::Image2D>()
                                          : in->data<float, cl::Image2D>(),
                                      im_w,
                                      im_h,
                                      cl_image2d_row_pitch,
                                      cl_image2d_slice_pitch,
                                      IoDirection::DtoH);
          folder_convertor.ImageToNCHW(
              in_data_v, real_out_v.data(), image_shape, in_dims);
          CHECK(real_out_v.size() == in->numel());
          *mean = compute_mean<float>(real_out_v.data(), real_out_v.size());
          *std_dev = compute_standard_deviation<float>(
              real_out_v.data(), in->numel(), true, *mean);
          *ave_grow_rate = compute_average_grow_rate<float>(real_out_v.data(),
                                                            real_out_v.size());
          std::shared_ptr<lite::Tensor> real_out_t(new lite::Tensor);
          real_out_t->Resize(in_dims);
          float* real_out_data = real_out_t->mutable_data<float>();
          memcpy(real_out_data,
                 real_out_v.data(),
                 real_out_v.size() * sizeof(float));
          if (write_result_to_file) {
            write_tensorfile<float>(real_out_t.get(), name, log_dir_);
          }
          return;
        }
        case DATALAYOUT(kNCHW): {
          auto* in_data_v =
              use_fp16
                  ? static_cast<void*>(calloc(in->numel(), sizeof(uint16_t)))
                  : static_cast<void*>(calloc(in->numel(), sizeof(float)));
          std::vector<float> real_out_v(in->numel());
          TargetWrapperCL::MemcpySync(
              in_data_v,
              use_fp16 ? in->data<half_t, cl::Buffer>()
                       : in->data<float, cl::Buffer>(),
              in->numel() * (use_fp16 ? sizeof(uint16_t) : sizeof(float)),
              IoDirection::DtoH);
          VLOG(1) << name << ":" << in->numel();
          if (use_fp16) {
            HalfArray2FloatArray(static_cast<half_t*>(in_data_v),
                                 real_out_v.data(),
                                 in->numel());
          } else {
            memcpy(real_out_v.data(), in_data_v, in->numel() * sizeof(float));
          }
          *mean = compute_mean<float>(real_out_v.data(), real_out_v.size());
          *std_dev = compute_standard_deviation<float>(
              real_out_v.data(), in->numel(), true, *mean);
          *ave_grow_rate = compute_average_grow_rate<float>(real_out_v.data(),
                                                            real_out_v.size());
          std::shared_ptr<lite::Tensor> real_out_t(new lite::Tensor);
          real_out_t->Resize(in->dims());
          float* real_out_data = real_out_t->mutable_data<float>();
          memcpy(real_out_data,
                 real_out_v.data(),
                 real_out_v.size() * sizeof(float));
          if (write_result_to_file) {
            write_tensorfile<float>(real_out_t.get(), name, log_dir_);
          }
          return;
        }
        default:
          *mean = -222222222222;
          *std_dev = -22222222222;
          *ave_grow_rate = -22222222222;
          LOG(ERROR) << unsupported_error_log;
          return;
      }
#endif
#ifdef LITE_WITH_CUDA
    } else if (target_type == TARGET(kCUDA)) {
      switch (precision_type) {
        case PRECISION(kAny):
        case PRECISION(kFloat): {
          std::vector<float> in_data_v(in->numel(), 0);
          TargetWrapperCuda::MemcpySync(in_data_v.data(),
                                        in->data<float>(),
                                        in->numel() * sizeof(float),
                                        IoDirection::DtoH);
          VLOG(1) << name << ":" << in->numel();
          *mean = compute_mean<float>(in_data_v.data(), in->numel());
          *std_dev = compute_standard_deviation<float>(
              in_data_v.data(), in->numel(), true, *mean);
          *ave_grow_rate =
              compute_average_grow_rate<float>(in_data_v.data(), in->numel());
          if (write_result_to_file) {
            write_tensorfile<float>(in, name, log_dir_);
          }
          return;
        }
        case PRECISION(kInt32): {
          std::vector<int> in_data_v(in->numel(), 0);
          TargetWrapperCuda::MemcpySync(in_data_v.data(),
                                        in->data<int>(),
                                        in->numel() * sizeof(int),
                                        IoDirection::DtoH);
          VLOG(1) << name << ":" << in->numel();
          *mean = compute_mean<int>(in_data_v.data(), in->numel());
          *std_dev = compute_standard_deviation<int>(
              in_data_v.data(), in->numel(), true, *mean);
          *ave_grow_rate =
              compute_average_grow_rate<int>(in_data_v.data(), in->numel());
          if (write_result_to_file) {
            write_tensorfile<int>(in, name, log_dir_);
          }
          return;
        }
        case PRECISION(kInt64): {
          std::vector<int64_t> in_data_v(in->numel(), 0);
          TargetWrapperCuda::MemcpySync(in_data_v.data(),
                                        in->data<int64_t>(),
                                        in->numel() * sizeof(int64_t),
                                        IoDirection::DtoH);
          VLOG(1) << name << ":" << in->numel();
          *mean = compute_mean<int64_t>(in_data_v.data(), in->numel());
          *std_dev = compute_standard_deviation<int64_t>(
              in_data_v.data(), in->numel(), true, *mean);
          *ave_grow_rate =
              compute_average_grow_rate<int64_t>(in_data_v.data(), in->numel());
          if (write_result_to_file) {
            write_tensorfile<int64_t>(in, name, log_dir_);
          }
          return;
        }
        case PRECISION(kFP16): {
          std::vector<float> in_data_v(in->numel(), 0);
          lite::Tensor fp32_tensor;
          fp32_tensor.Resize(in->dims());
          lite::cuda::math::fp16_to_fp32(
              in->numel(),
              in->data<half>(),
              fp32_tensor.mutable_data<float>(TARGET(kCUDA)));
          TargetWrapperCuda::MemcpySync(in_data_v.data(),
                                        fp32_tensor.data<float>(),
                                        in->numel() * sizeof(float),
                                        IoDirection::DtoH);
          VLOG(1) << name << ":" << in->numel();
          *mean = compute_mean<float>(in_data_v.data(), in->numel());
          *std_dev = compute_standard_deviation<float>(
              in_data_v.data(), in->numel(), true, *mean);
          *ave_grow_rate =
              compute_average_grow_rate<float>(in_data_v.data(), in->numel());
          if (write_result_to_file) {
            write_tensorfile<float>(in, name, log_dir_);
          }
          return;
        }
        default:
          *mean = -222222222222;
          *std_dev = -22222222222;
          *ave_grow_rate = -22222222222;
          LOG(ERROR) << unsupported_error_log;
          return;
      }
#endif
    } else {
      *mean = -111111111111;
      *std_dev = -11111111111;
      *ave_grow_rate = -11111111111;
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
            << DataLayoutToStr(inst->kernel()->layout())
            << ", write_result_to_file_:" << write_result_to_file_;

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
        auto tmp = op_scope->FindVar(out_name);
        if (tmp->IsType<Tensor>()) {
          const Tensor* tout =
              op_scope->FindVar(out_name)->GetMutable<Tensor>();
          double mean = -999999;
          double std_dev = -100000;
          double ave_grow_rate = 99999;
          std::string mean_str{"unused"};
          std::string std_dev_str{"unused"};
          std::string ave_grow_rate_str{"unused"};
          std::string new_out_name = rename_out_for_mem_reuse_pass(out_name);

          if (tout->IsInitialized()) {
            compute_tensor_precision_info(tout,
                                          op_name,
                                          type->layout(),
                                          &mean,
                                          &std_dev,
                                          &ave_grow_rate,
                                          new_out_name,
                                          write_result_to_file_);
            mean_str = std::to_string(mean);
            std_dev_str = std::to_string(std_dev);
            ave_grow_rate_str = std::to_string(ave_grow_rate);
          } else {
            LOG(INFO) << out_name << " is not inited.";
          }
          std::string kernel_info = op_name + ":" + kernel_place;
          std::string output_arg_info = new_out_name + ":" +
                                        TargetToStr(type->target()) + "/" +
                                        PrecisionToStr(type->precision()) +
                                        "/" + DataLayoutToStr(type->layout());

          ss << setw(45) << left << kernel_info << " " << setw(70) << left
             << output_arg_info << " " << setw(15) << left << tout->dims()
             << " " << setw(15) << left << mean_str << " " << setw(15) << left
             << std_dev_str << " " << setw(15) << left << ave_grow_rate_str
             << std::endl;
        } else if (tmp->IsType<std::vector<Tensor>>()) {
          auto touts =
              op_scope->FindVar(out_name)->GetMutable<std::vector<Tensor>>();
          for (auto t : *touts) {
            const Tensor* tout = &t;
            double mean = -999999;
            double std_dev = -100000;
            double ave_grow_rate = 99999;
            std::string mean_str{"unused"};
            std::string std_dev_str{"unused"};
            std::string ave_grow_rate_str{"unused"};
            std::string new_out_name = rename_out_for_mem_reuse_pass(out_name);

            if (tout->IsInitialized()) {
              compute_tensor_precision_info(tout,
                                            op_name,
                                            type->layout(),
                                            &mean,
                                            &std_dev,
                                            &ave_grow_rate,
                                            new_out_name,
                                            write_result_to_file_);
              mean_str = std::to_string(mean);
              std_dev_str = std::to_string(std_dev);
              ave_grow_rate_str = std::to_string(ave_grow_rate);
            } else {
              LOG(INFO) << out_name << " is not inited.";
            }
            std::string kernel_info = op_name + ":" + kernel_place;
            std::string output_arg_info = new_out_name + ":" +
                                          TargetToStr(type->target()) + "/" +
                                          PrecisionToStr(type->precision()) +
                                          "/" + DataLayoutToStr(type->layout());

            ss << setw(45) << left << kernel_info << " " << setw(70) << left
               << output_arg_info << " " << setw(15) << left << tout->dims()
               << " " << setw(15) << left << mean_str << " " << setw(15) << left
               << std_dev_str << " " << setw(15) << left << ave_grow_rate_str
               << std::endl;
          }
        }
      }
    }
    write_precision_summary_tofile(ss.str(), summary_log_dir_);
    return ss.str();
  }

 private:
#ifdef LITE_WITH_ANDROID
  std::string log_dir_{"/storage/emulated/0/PaddleLite_" + get_date_str() +
                       "/"};
#elif defined(_MSC_VER)
  std::string log_dir_{"C:/PaddleLite_" + get_date_str() + "/"};
#else
  std::string log_dir_{"/tmp/PaddleLite_" + get_date_str() + "/"};
#endif
  std::string summary_log_dir_{log_dir_ + "precision_summary.log"};
  std::map<std::string, size_t> out_tensor_names_map;
  bool write_result_to_file_{false};
};

}  // namespace profile
}  // namespace lite
}  // namespace paddle

// TODO(ysh329): need to remove.
// keep this method only for arm/math/conditional_block_compute
#define LITE_PRECISION_PROFILE(inst) \
  { auto a = paddle::lite::profile::PrecisionProfiler(&inst); }
