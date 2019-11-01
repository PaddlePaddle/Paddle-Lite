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

#include <vector>
#include "lite/backends/x86/math/softmax.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

static inline int CanonicalAxis(const int axis, const int rank) {
  if (axis < 0) {
    return axis + rank;
  }
  return axis;
}

static inline int SizeToAxis(const int axis, lite::DDim dims) {
  int size = 1;
  for (int i = 0; i < axis; i++) {
    size *= dims[i];
  }
  return size;
}

static inline int SizeFromAxis(const int axis, lite::DDim dims) {
  int size = 1;
  for (size_t i = axis; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T>
class SoftmaxCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  using param_t = operators::SoftmaxParam;

  void Run() override {
    auto& param = *param_.get_mutable<operators::SoftmaxParam>();
    auto& context = ctx_->As<X86Context>();
    CHECK(param.output);
    CHECK(param.x);
    param.output->mutable_data<T>();
    const int rank = param.x->dims().size();
    const int axis = CanonicalAxis(param.axis, rank);
    int axis_dim = param.x->dims()[axis];
    const int n = SizeToAxis(axis, param.x->dims());
    const int d = SizeFromAxis(axis, param.x->dims());
    std::vector<int64_t> shape{n, d};

    lite::Tensor input_2d, out_2d;
    input_2d.ShareDataWith(*param.x);
    input_2d.Resize(lite::DDim(shape));
    out_2d.ShareDataWith(*param.output);
    out_2d.Resize(lite::DDim(shape));

    lite::x86::math::SoftmaxFunctor<lite::TargetType::kX86, T, true>()(
        context, axis_dim, &input_2d, &out_2d);
  }

  virtual ~SoftmaxCompute() = default;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
