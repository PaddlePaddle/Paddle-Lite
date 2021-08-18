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

#pragma once

#include "lite/backends/x86/fluid/data_type.h"
#include "lite/backends/x86/fluid/hostdevice.h"
#include "lite/backends/x86/fluid/transform.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <typename InT, typename OutT>
struct CastOpTransformFunctor {
  HOSTDEVICE OutT operator()(InT in) const { return static_cast<OutT>(in); }
};

template <lite::TargetType Target, typename InT>
class CastOpFunctor {
 public:
  CastOpFunctor(const lite::Tensor* in,
                lite::Tensor* out,
                const lite::Context<Target>& context)
      : input(in), output(out), ctx(context) {}

  template <typename OutT>
  void apply() const {
    auto* in_begin = input->data<InT>();
    auto numel = input->dims().production();
    auto* in_end = in_begin + numel;
    auto* out_begin = output->mutable_data<OutT>();
    paddle::lite::fluid::Transform<lite::TargetType::kX86> trans;
    trans(
        ctx, in_begin, in_end, out_begin, CastOpTransformFunctor<InT, OutT>());
  }

 private:
  const lite::Tensor* input;
  lite::Tensor* output;
  const lite::Context<Target>& ctx;
};

template <typename InT>
class CastCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::CastParam;

  void Run() override {
    auto param = param_.get_mutable<param_t>();
    auto& context = ctx_->As<X86Context>();
    auto x = param->X;
    auto out = param->Out;
    auto out_dtype = param->out_dtype;
    paddle::lite::fluid::VisitDataType(
        static_cast<framework::proto::VarType::Type>(out_dtype),
        CastOpFunctor<lite::TargetType::kX86, InT>(x, out, context));
  }
  virtual ~CastCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
