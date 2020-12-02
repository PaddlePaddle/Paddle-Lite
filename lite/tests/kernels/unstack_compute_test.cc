// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

template <typename dtype>
void SplitComputeRef(const Tensor* x,
                     const std::vector<Tensor*>& outs,
                     int axis) {
  const dtype* din = x->data<dtype>();
  auto in_dim = x->dims();
  std::vector<int> in_strides(in_dim.size());
  in_strides[in_dim.size() - 1] = in_dim[in_dim.size() - 1];
  for (int i = in_dim.size() - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * in_dim[i];
  }
  if (axis < 0) {
    axis += in_dim.size();
  }

  int input_offset = 0;
  for (auto out : outs) {
    auto out_dim = out->dims().Vectorize();
    out_dim.insert(out_dim.begin() + axis, 1);
    std::vector<int> out_strides(out_dim.size());
    out_strides[out_dim.size() - 1] = out_dim[out_dim.size() - 1];
    for (int i = out_dim.size() - 2; i >= 0; --i) {
      out_strides[i] = out_strides[i + 1] * out_dim[i];
    }

    dtype* out_data = out->mutable_data<dtype>();
    int before = out_strides[0] / out_strides[axis];
    int in_after = in_strides[axis];
    int out_after = out_strides[axis];

    for (int i = 0; i < before; ++i) {
      std::memcpy(out_data + i * out_after,
                  din + input_offset + i * in_after,
                  sizeof(dtype) * out_after);
    }
    input_offset += out_strides[axis];
  }
}

template <class T>
class UnstackComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::vector<std::string> outs_;
  DDim x_dims_{{1, 5, 6, 7}};
  int axis_;
  int num_;

 public:
  UnstackComputeTester(const Place& place,
                       const std::string& alias,
                       DDim x_dims,
                       int axis = 0)
      : TestCase(place, alias), x_dims_(x_dims), axis_(axis) {
    if (axis < 0) {
      axis += x_dims.size();
    }
    num_ = x_dims[axis];
    for (int i = 0; i < num_; i++) {
      outs_.emplace_back("out_" + std::to_string(i));
    }
  }

  void RunBaseline(Scope* scope) override {
    auto out_shape = x_dims_.Vectorize();
    int axis = axis_ >= 0 ? axis_ : axis_ + x_dims_.size();
    out_shape.erase(out_shape.begin() + axis);
    std::vector<Tensor*> outs;
    for (auto out_name : outs_) {
      auto out = scope->NewTensor(out_name);
      out->Resize(out_shape);
      outs.emplace_back(out);
    }

    SplitComputeRef<T>(scope->FindTensor(x_), outs, axis);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("unstack");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Y", outs_);
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("num", num_);
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    fill_data_rand<T>(x_data.data(), -1, 1, x_data.size());
    SetCommonTensor<T>(x_, x_dims_, x_data.data());
  }
};

template <class T = float>
void TestUnstack(Place place,
                 float abs_error,
                 const std::vector<int64_t>& x_shape) {
  place.precision = lite_api::PrecisionTypeTrait<T>::Type();
  std::vector<int> axes;
  int shape_size = x_shape.size();
  for (int i = -shape_size; i < shape_size; i++) {
    axes.emplace_back(i);
  }
  for (int axis : axes) {
    std::unique_ptr<arena::TestCase> tester(
        new UnstackComputeTester<T>(place, "def", DDim(x_shape), axis));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision();
  }
}

TEST(unstack, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_XPU) && (not defined(LITE_WITH_XTCL))
  place = TARGET(kXPU);
#elif defined(LITE_WITH_X86) || defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#else
  return;
#endif

  TestUnstack<float>(place, abs_error, {2, 3, 4, 5});
  TestUnstack<float>(place, abs_error, {1, 3, 4});
  TestUnstack<float>(place, abs_error, {4});
}

}  // namespace lite
}  // namespace paddle
