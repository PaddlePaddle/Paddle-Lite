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
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

std::vector<int> CalStrides(const DDim& dims) {
  int dsize = dims.size();
  std::vector<int> strides(dsize, 1);
  for (int i = dsize - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

std::vector<int> CalIndex(const std::vector<int>& strides, int offset) {
  int dsize = strides.size();
  std::vector<int> index(dsize, 0);
  for (int i = 0; i < dsize; i++) {
    index[i] = offset / strides[i];
    offset %= strides[i];
  }
  return index;
}

std::vector<int> TransIndex(const std::vector<int>& in_index,
                            const std::vector<int>& axis) {
  std::vector<int> out_index(in_index.size(), 0);
  for (int i = 0; i < axis.size(); i++) {
    out_index[i] = in_index[axis[i]];
  }
  return out_index;
}

int CalOffset(const std::vector<int>& strides, const std::vector<int>& index) {
  int offset = 0;
  for (int i = 0; i < index.size(); i++) {
    offset += strides[i] * index[i];
  }
  return offset;
}

class TransposeComputeTester : public arena::TestCase {
 protected:
// common attributes for this op.
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  std::string op_type_ = "transpose";
#else
  std::string op_type_ = "transpose2";
#endif
  std::string input_ = "x";
  std::string output_ = "out";
  std::string xshape_ = "xshape";
  DDim dims_;
  std::vector<int> axis_;

 public:
  TransposeComputeTester(const Place& place,
                         const std::string& alias,
                         DDim dims,
                         std::vector<int> axis)
      : TestCase(place, alias), dims_(dims), axis_(axis) {}

  void RunBaseline(Scope* scope) override {
    auto* out = scope->NewTensor(output_);
    CHECK(out);

    auto* x = scope->FindTensor(input_);

    std::vector<int64_t> out_shape(dims_.size(), 0);
    for (size_t i = 0; i < dims_.size(); i++) {
      out_shape[i] = dims_[axis_[i]];
    }
    out->Resize(out_shape);
    auto out_dims = out->dims();

    std::vector<int> x_strides = CalStrides(dims_);
    std::vector<int> out_strides = CalStrides(out_dims);

    auto x_data = x->data<float>();
    auto out_data = out->mutable_data<float>();

    for (int i = 0; i < dims_.production(); i++) {
      std::vector<int> x_index = CalIndex(x_strides, i);
      std::vector<int> out_index = TransIndex(x_index, axis_);
      int out_offset = CalOffset(out_strides, out_index);
      out_data[out_offset] = x_data[i];
    }

    if (op_type_ == "transpose2") {
      auto* xshape = scope->NewTensor(xshape_);
      auto xshape_dims = dims_.Vectorize();
      xshape_dims.insert(xshape_dims.begin(), 0);
      xshape->Resize(xshape_dims);
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    if (op_type_ == "transpose2") {
      op_desc->SetOutput("XShape", {xshape_});
    }
    op_desc->SetAttr("axis", axis_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(input_, dims_, din.data());
  }
};

void TestTranspose2D(Place place, float abs_error) {
  DDim x_dims{{4, 5}};
  std::vector<std::vector<int>> axes {
#if !defined(LITE_WITH_XPU)
    {0, 1},
#endif
        {1, 0},
  };
  for (auto axis : axes) {
    std::unique_ptr<arena::TestCase> tester(
        new TransposeComputeTester(place, "def", x_dims, axis));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision({"xshape"});
  }
}

void TestTranspose3D(Place place, float abs_error) {
  DDim x_dims{{3, 4, 5}};
  std::vector<std::vector<int>> axes {
#if !defined(LITE_WITH_XPU)
    {0, 1, 2},
#endif
        {0, 2, 1}, {1, 0, 2}, {2, 1, 0},
  };
  for (auto axis : axes) {
    std::unique_ptr<arena::TestCase> tester(
        new TransposeComputeTester(place, "def", x_dims, axis));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision({"xshape"});
  }
}

#ifdef ENABLE_ARM_FP16
void TestTranspose4D_fp16(Place place, float abs_error) {
  DDim x_dims{{1, 12, 19, 19}};
  std::vector<std::vector<int>> axes{
      {0, 2, 3, 1}, {0, 3, 1, 2},
  };
  for (auto axis : axes) {
    std::unique_ptr<arena::TestCase> tester(
        new TransposeComputeTester(place, "def", x_dims, axis));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision({"xshape"});
  }
}

void TestTranspose3D_fp16(Place place, float abs_error) {
  DDim x_dims{{1, 1917, 21}};
  std::vector<std::vector<int>> axes{
      {0, 2, 1}, {1, 0, 2}, {2, 1, 0},
  };
  for (auto axis : axes) {
    std::unique_ptr<arena::TestCase> tester(
        new TransposeComputeTester(place, "def", x_dims, axis));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision({"xshape"});
  }
}

TEST(Transpose_fp16, precision) {
  float abs_error = 2e-5;
  Place place1(TARGET(kARM), PRECISION(kFP16));
  TestTranspose4D_fp16(place1, abs_error);
  TestTranspose3D_fp16(place1, abs_error);
}

#endif

void TestTranspose4D(Place place, float abs_error) {
  DDim x_dims{{2, 3, 4, 5}};
  std::vector<std::vector<int>> axes {
#if !defined(LITE_WITH_XPU)
    {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {3, 1, 2, 0}, {3, 1, 0, 2},
#endif
#if !defined(LITE_WITH_NPU)
        {0, 2, 3, 1}, {0, 3, 1, 2},
#endif
  };
  for (auto axis : axes) {
    std::unique_ptr<arena::TestCase> tester(
        new TransposeComputeTester(place, "def", x_dims, axis));
    arena::Arena arena(std::move(tester), place, abs_error);
    arena.TestPrecision({"xshape"});
  }
}

TEST(Transpose, precision) {
  float abs_error = 2e-5;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-2;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // Using fp16 in NPU
#elif defined(LITE_WITH_OPENCL)
  place = Place(TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  abs_error = 1e-2;  // Using fp16 in OPENCL
#elif defined(LITE_WITH_ARM)
  place = TARGET(kARM);
#else
  return;
#endif

  TestTranspose2D(place, abs_error);
  TestTranspose3D(place, abs_error);
  TestTranspose4D(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
