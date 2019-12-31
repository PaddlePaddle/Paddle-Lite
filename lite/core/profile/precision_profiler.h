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
#include <unordered_map>
#include <vector>
#include "lite/core/program.h"

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
  using TargetW = TargetWrapper<TARGET(kCUDA)>;

  explicit PrecisionProfiler(const Instruction* inst,
                             const KernelContext* context)
      : inst_(inst), ctx_(context) {}
  ~PrecisionProfiler() {
    LOG(INFO) << ">> Running kernel: " << inst_->op()->op_info()->Repr()
              << " on Target " << TargetToStr(inst_->kernel()->target()) << " "
              << PrecisionToStr(inst_->kernel()->precision());
    auto tensor_mean = [this](const Tensor* in,
                              TargetType ttype,
                              PrecisionType ptype,
                              int64_t skip_num = 1,
                              std::string name = "inst") -> double {
      if (!in->data<int8_t>()) {
        return -99999;
      }
      double sum = 0.;

#define MEAN_VALUE(type, skip_num)                                            \
  if (ttype == TARGET(kCUDA)) {                                               \
    auto& ctx = const_cast<KernelContext*>(ctx_)->template As<CUDAContext>(); \
    auto stream = ctx.exec_stream();                                          \
    auto ptr = in->data<type>();                                              \
    int64_t num = in->numel();                                                \
    std::vector<type> output(num, 0);                                         \
    TargetW::MemcpyAsync(                                                     \
        output.data(), ptr, num * sizeof(type), IoDirection::DtoH, stream);   \
    cudaStreamSynchronize(stream);                                            \
    for (int64_t i = 0; i < num; i += skip_num) {                             \
      sum += output[i];                                                       \
    }                                                                         \
    return sum / num / skip_num;                                              \
  } else {                                                                    \
    auto ptr = in->data<type>();                                              \
    for (int64_t i = 0; i < in->numel(); i += skip_num) {                     \
      sum += ptr[i];                                                          \
    }                                                                         \
    return sum / in->numel() / skip_num;                                      \
  }

      switch (ptype) {
        case PRECISION(kFloat): {
          MEAN_VALUE(float, skip_num);
        }
        case PRECISION(kAny): {
          MEAN_VALUE(float, skip_num);
        }
        case PRECISION(kInt8): {
          MEAN_VALUE(int8_t, skip_num);
        }
        case PRECISION(kInt32): {
          MEAN_VALUE(int32_t, skip_num);
        }
        case PRECISION(kInt64): {
          MEAN_VALUE(int64_t, skip_num);
        }
        default:
          LOG(INFO) << "unsupport data type: " << PrecisionToStr(ptype);
          return 0.;
      }
#undef MEAN_VALUE
    };
    if (inst_->op()->op_info()->Type() != "fetch") {
      auto op = const_cast<lite::OpLite*>(inst_->op());
      auto kernel = inst_->kernel();
      auto op_scope = op->scope();
      auto out_names = op->op_info()->output_names();
      for (auto& out_name : out_names) {
        std::string out_arg_name;
        op->op_info()->GetOutputArgname(out_name, &out_arg_name);
        auto type = kernel->GetOutputDeclType(out_arg_name);

        if (type->IsTensor()) {
          auto tout = op_scope->FindVar(out_name)->GetMutable<Tensor>();
          double mean =
              tensor_mean(tout, type->target(), type->precision(), 1, out_name);
          double per100 =
              tensor_mean(tout, type->target(), type->precision(), 2, out_name);
          LoD lod_info = tout->lod();
          std::ostringstream oss;
          oss << "[";
          for (size_t i = 0; i < lod_info.size(); ++i) {
            oss << "[";
            for (size_t j = 0; j < lod_info[i].size(); ++j) {
              oss << std::to_string(lod_info[i][j]) << " ";
            }
            oss << "], ";
          }
          oss << "]";
          LOG(INFO) << "output name: " << out_name << ", dims: " << tout->dims()
                    << ", precision: " << PrecisionToStr(type->precision())
                    << ", mean value: " << mean << " shape:" << tout->dims()
                    << ", per 100: " << per100 << ", lod is " << oss.str();
        } else if (type->IsTensorList()) {
          auto tout =
              op_scope->FindVar(out_name)->GetMutable<std::vector<Tensor>>();
          for (auto& t : *tout) {
            double mean =
                tensor_mean(&t, type->target(), type->precision(), 1, out_name);
            LOG(INFO) << "output name: " << out_name << ", dims: " << t.dims()
                      << ", precision: " << PrecisionToStr(type->precision())
                      << ", mean value: " << mean;
          }
        }
      }
    }
  }

 private:
  const Instruction* inst_{nullptr};
  const KernelContext* ctx_{nullptr};
};

}  // namespace profile
}  // namespace lite
}  // namespace paddle

#define LITE_PRECISION_PROFILE(inst, ctx) \
  { auto a = paddle::lite::profile::PrecisionProfiler(&inst, ctx); }
