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
  explicit PrecisionProfiler(const Instruction* inst) : inst_(inst) {}
  ~PrecisionProfiler() {
    LOG(INFO) << ">> Running kernel: " << inst_->op()->op_info()->Repr()
              << " on Target " << TargetToStr(inst_->kernel()->target()) << " "
              << PrecisionToStr(inst_->kernel()->precision());
    auto tensor_mean = [](const Tensor* in,
                          PrecisionType ptype,
                          std::string name = "inst") -> double {
      if (!in->data<int8_t>()) {
        return -99999;
      }
      double sum = 0.;
      switch (ptype) {
        case PRECISION(kFloat): {
          auto ptr = in->data<float>();
          // write_tensorfile<float>(in, name);
          for (int i = 0; i < in->numel(); ++i) {
            sum += ptr[i];
          }
          return sum / in->numel();
        }
        case PRECISION(kAny): {
          auto ptr = in->data<float>();
          // write_tensorfile<float>(in, name);
          for (int i = 0; i < in->numel(); ++i) {
            sum += ptr[i];
          }
          return sum / in->numel();
        }
        case PRECISION(kInt8): {
          auto ptr = in->data<int8_t>();
          // write_tensorfile<int8_t>(in, name);
          for (int i = 0; i < in->numel(); ++i) {
            sum += ptr[i];
          }
          return sum / in->numel();
        }
        case PRECISION(kInt32): {
          auto ptr = in->data<int32_t>();
          // write_tensorfile<int32_t>(in, name);
          for (int i = 0; i < in->numel(); ++i) {
            sum += ptr[i];
          }
          return sum / in->numel();
        }
        default:
          LOG(INFO) << "unsupport data type: " << PrecisionToStr(ptype);
          return 0.;
      }
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
          double mean = tensor_mean(tout, type->precision(), out_name);
          LOG(INFO) << "output name: " << out_name << ", dims: " << tout->dims()
                    << ", precision: " << PrecisionToStr(type->precision())
                    << ", mean value: " << mean << " shape:" << tout->dims();
        } else if (type->IsTensorList()) {
          auto tout =
              op_scope->FindVar(out_name)->GetMutable<std::vector<Tensor>>();
          for (auto& t : *tout) {
            double mean = tensor_mean(&t, type->precision(), out_name);
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
};

}  // namespace profile
}  // namespace lite
}  // namespace paddle

#define LITE_PRECISION_PROFILE(inst) \
  { auto a = paddle::lite::profile::PrecisionProfiler(&inst); }
