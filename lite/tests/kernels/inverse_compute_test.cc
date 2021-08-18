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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"

namespace paddle {
namespace lite {

class InverseComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  std::string alias_ = "fp32";
  DDim dims_{{5, 5, 5, 5}};

 public:
  InverseComputeTester(
      const Place& place, const std::string& alias, int n, int c, int h, int w)
      : TestCase(place, alias), alias_(alias) {
    dims_ = DDim(std::vector<int64_t>({n, c, h, w}));
  }

  template <typename T>
  void MulAdd(T* x, int i, int j, T a, int cols) {
    for (int k = 0; k < cols; k++) x[j * cols + k] += (a * x[i * cols + k]);
  }
  template <typename T>
  void swap2row(T* x, int cols, int i, int j) {
    for (int k = 0; k < cols; k++) {
      T temp = x[i * cols + k];
      x[i * cols + k] = x[j * cols + k];
      x[j * cols + k] = temp;
    }
  }
  template <typename T>
  void GaussInv(const T* x, T* out, int rows, int cols) {
    T* temp = reinterpret_cast<T*>(
        TargetMalloc(TARGET(kHost), 2 * rows * cols * sizeof(T)));
    for (int i = 0; i < 2 * rows * cols; i++) temp[i] = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) temp[i * 2 * cols + j] = x[i * cols + j];
      temp[i * 2 * cols + cols + i] = 1;
    }
    for (int i = 0; i < rows; i++) {
      T a1 = temp[i * 2 * cols + i];
      int swap_i = i;
      for (int k = i + 1; k < rows; k++)
        if (std::abs(temp[k * 2 * cols + i] > std::abs(a1))) {
          swap_i = k;
          a1 = temp[k * 2 * cols + i];
        }
      if (swap_i != i) swap2row(temp, 2 * cols, swap_i, i);
      for (int k = i; k < 2 * cols; k++) temp[i * 2 * cols + k] /= a1;
      for (int j = 0; j < i; j++) {
        T a2 = temp[j * 2 * cols + i];
        MulAdd(temp, i, j, -a2, 2 * cols);
      }
      for (int j = i + 1; j < rows; j++) {
        T a2 = temp[j * 2 * cols + i];
        MulAdd(temp, i, j, -a2, 2 * cols);
      }
    }

    for (int i = 0; i < rows; i++)
      for (int j = 0; j < cols; j++)
        out[i * cols + j] = temp[i * 2 * cols + j + cols];
    TargetFree(TARGET(kHost), temp);
  }

  template <typename indtype>
  void RunBaselineKernel(Scope* scope) {
    auto* out = scope->NewTensor(output_);
    CHECK(out);
    int64_t nchw[] = {dims_[0], dims_[1], dims_[2], dims_[3]};
    std::vector<int64_t> output_shape(nchw, nchw + 4);

    DDim output_dims(output_shape);
    out->Resize(output_dims);
    auto* output_data = out->mutable_data<indtype>();
    auto* x = scope->FindTensor(input_);
    const auto* x_data = x->data<indtype>();
    int rows = dims_[2];
    int cols = dims_[3];
    int batch_size = output_dims.count(0, 2);

    for (int i = 0; i < batch_size; i++)
      GaussInv(
          x_data + i * rows * cols, output_data + i * rows * cols, rows, cols);
  }

  void RunBaseline(Scope* scope) override {
    if (alias_ == "fp32") {
      RunBaselineKernel<float>(scope);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) override {
    op_desc->SetType("inverse");
    op_desc->SetInput("Input", {input_});
    op_desc->SetOutput("Output", {output_});
  }
  void PrepareData() override {
    if (alias_ == "fp32") {
      std::vector<float> data(dims_.production());
      for (int i = 0; i < dims_.production(); i++) {
        float sign = i % 3 == 0 ? -1.0f : 1.0f;
        data[i] = sign * static_cast<float>(i % 128) * 0.13f + 1.0f;
      }
      SetCommonTensor(input_, dims_, data.data());
    }
  }
};

void TestInverse(const Place& place) {
  for (int n : {5}) {
    for (int c : {5}) {
      for (int h : {5}) {
        for (int w : {5}) {
          std::vector<std::string> alias_vec{"fp32"};
          for (std::string alias : alias_vec) {
            std::unique_ptr<arena::TestCase> tester(
                new InverseComputeTester(place, alias, n, c, h, w));
            arena::Arena arena(std::move(tester), place, 0.1);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

TEST(Inverse, precision) {
  Place place;
#if defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#elif defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif
  TestInverse(place);
}

}  // namespace lite
}  // namespace paddle
