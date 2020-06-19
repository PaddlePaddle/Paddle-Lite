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

#include <Eigen/Core>
#include <string>
#include <vector>
#include "lite/backends/x86/math/math_function.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/operators/layout_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

template <paddle::lite_api::PrecisionType>
struct FPTypeTraits {};

template <>
struct FPTypeTraits<paddle::lite_api::PrecisionType::kFloat> {
  typedef float T;
};

template <>
struct FPTypeTraits<paddle::lite_api::PrecisionType::kFP16> {
  typedef paddle::lite::fluid::float16 T;
};

template <>
struct FPTypeTraits<paddle::lite_api::PrecisionType::kInt8> {
  typedef int8_t T;
};

template <lite::TargetType Target, typename T>
inline void LayoutTransCompute(const int dim,
                               const lite::Context<Target>& context,
                               const lite::Tensor& in,
                               lite::Tensor* out,
                               const std::vector<int>& axis) {
  switch (dim) {
    case 2:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 2> trans2;
      trans2(context, in, out, axis);
      break;
    case 3:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 3> trans3;
      trans3(context, in, out, axis);
      break;
    case 4:
      paddle::lite::x86::math::Transpose<lite::TargetType::kX86, T, 4> trans4;
      trans4(context, in, out, axis);
      break;
    default:
      CHECK(0) << ("Unsupport dim in mlu layout");
  }
}

template <PrecisionType Precision>
class LayoutNchwToNhwcCompute
    : public KernelLite<TARGET(kMLU), Precision, DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::LayoutParam;

  void Run() override {
    auto& param = this->template Param<param_t>();
    auto* x = param.x;
    auto* out = param.y;
    out->template mutable_data<typename FPTypeTraits<Precision>::T>();
    auto x_dims = param.x->dims().size();
    auto& context = this->ctx_->template As<X86Context>();

    const auto origin_dims = out->dims().Vectorize();

    std::vector<int> axis;
    switch (x_dims) {
      case 2:
        axis = {0, 1};
        break;
      case 3:
        axis = {0, 2, 1};
        out->Resize(std::vector<int64_t>{
            out->dims()[0], out->dims()[2], out->dims()[1]});
        break;
      case 4:
        axis = {0, 2, 3, 1};
        out->Resize(std::vector<int64_t>{
            out->dims()[0], out->dims()[2], out->dims()[3], out->dims()[1]});
        break;
      default:
        CHECK(0) << "Unsupport dim in mlu layout nchw to nhwc";
    }

    LayoutTransCompute<lite::TargetType::kX86,
                       typename FPTypeTraits<Precision>::T>(
        x_dims, context, *x, out, axis);

    if (x_dims > 2) {
      out->Resize(origin_dims);
    }
  }

  std::string doc() const override {
    return "Mlu layout transform nchw to nhwc";
  }
};

template <PrecisionType Precision>
class LayoutNhwcToNchwCompute
    : public KernelLite<TARGET(kMLU), Precision, DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::LayoutParam;

  void Run() override {
    auto& param = this->template Param<param_t>();
    auto* x = param.x;
    auto* out = param.y;
    out->template mutable_data<typename FPTypeTraits<Precision>::T>();
    auto x_dims = param.x->dims().size();
    auto& context = this->ctx_->template As<X86Context>();

    const auto origin_dims = out->dims().Vectorize();

    std::vector<int> axis;
    switch (x_dims) {
      case 2:
        axis = {0, 1};
        break;
      case 3:
        out->Resize(std::vector<int64_t>{
            out->dims()[0], out->dims()[2], out->dims()[1]});
        axis = {0, 2, 1};
        break;
      case 4:
        out->Resize(std::vector<int64_t>{
            out->dims()[0], out->dims()[3], out->dims()[1], out->dims()[2]});
        axis = {0, 3, 1, 2};
        break;
      default:
        CHECK(0) << "Unsupport dim in mlu layout nhwc to nchw";
    }

    LayoutTransCompute<lite::TargetType::kX86,
                       typename FPTypeTraits<Precision>::T>(
        x_dims, context, *x, out, axis);

    if (x_dims > 2) {
      out->Resize(origin_dims);
    }
  }

  std::string doc() const override {
    return "Mlu layout transform nhwc to nchw";
  }
};

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
