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

template <class T = float, class R = int64_t, class A = int32_t>
class GatherComputeTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string op_type_ = "gather";
  std::string x_ = "x";
  std::string index_ = "index";
  std::string axis_ = "axis";
  std::string out_ = "out";
  DDim x_dims_{{5, 4, 2, 3}};
  DDim index_dims_{{2, 1}};
  DDim axis_dims_{{1}};

 public:
  GatherComputeTest(const Place& place,
                    const std::string& alias,
                    const DDim& x_dims,
                    const DDim& index_dims,
                    const DDim& axis_dims)
      : TestCase(place, alias),
        x_dims_(x_dims),
        index_dims_(index_dims),
        axis_dims_(axis_dims) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto index = scope->FindTensor(index_);
    auto axis = scope->FindTensor(axis_);
    auto x_dims = x->dims();
    auto index_dims = index->dims();
    CHECK(index_dims.size() == 1 ||
          (index_dims.size() == 2 && index_dims[1] == 1));
    CHECK_EQ(index_dims.size(), 1);
    if (axis_dims_.production() == 1) {
      auto* axis_data = axis->template data<A>();
      auto* index_data = index->template data<R>();
      auto* input_data = x->template data<T>();

      int index_size = index->numel();
      int input_size = x->numel();
      auto input_dim = x->dims();
      int axis_index = axis_data[0];
      int inner_dim_size = 1;
      int outer_dim_size = 1;
      std::vector<int64_t> out_dim_vec;
      for (int i = 0; i < axis_index; i++) {
        inner_dim_size *= input_dim[i];
        out_dim_vec.push_back(input_dim[i]);
      }
      out_dim_vec.push_back(index_size);
      for (int i = axis_index + 1; i < input_dim.size(); i++) {
        outer_dim_size *= input_dim[i];
        out_dim_vec.push_back(input_dim[i]);
      }
      auto out = scope->NewTensor(out_);
      CHECK(out);
      out->Resize(out_dim_vec);
      auto* out_data = out->template mutable_data<T>();

      int out_index = 0;
      for (int i = 0; i < inner_dim_size; i++) {
        for (int j = 0; j < index_size; j++) {
          for (int k = 0; k < outer_dim_size; k++) {
            int index = k + index_data[j] * outer_dim_size +
                        (i * input_size / inner_dim_size);
            out_data[out_index] = input_data[index];
            out_index++;
          }
        }
      }
      return;
    } else {
      auto out = scope->NewTensor(out_);
      CHECK(out);
      int batch_size = index_dims[0];
      DDim out_dims = x_dims;
      out_dims[0] = batch_size;
      out->Resize(out_dims);

      auto x_data = x->template data<T>();
      auto index_data = index->template data<R>();
      auto out_data = out->template mutable_data<T>();

      auto slice_num = x_dims[0];
      auto slice_size = x_dims.Slice(1, x_dims.size()).production();
      for (int i = 0; i < batch_size; i++) {
        auto index = index_data[i];
        CHECK_LT(index, slice_num) << "gather index[i] expected < " << slice_num
                                   << " but got " << index;
        CHECK_GE(index, 0) << "gather ids[i] expected >= 0 but got " << index;
        memcpy(out_data + i * slice_size,
               x_data + index * slice_size,
               slice_size * sizeof(T));
      }
      return;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType(op_type_);
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Index", {index_});
    if (axis_dims_.production() == 1) {
      op_desc->SetInput("Axis", {axis_});
    }
    op_desc->SetOutput("Out", {out_});
  }

  void PrepareData() override {
    std::vector<T> x(x_dims_.production());
    fill_data_rand(
        x.data(), static_cast<T>(-1), static_cast<T>(1), x_dims_.production());

    std::vector<R> index(index_dims_.production());
    fill_data_rand<R>(
        index.data(), 0, x_dims_[0] - 1, index_dims_.production());
    std::vector<A> axis(axis_dims_.production());
    if (axis_dims_.production() == 1) {
      fill_data_rand<A>(
          axis.data(), 0, x_dims_.size() - 1, axis_dims_.production());
    }
    SetCommonTensor(x_, x_dims_, x.data());
    SetCommonTensor(index_, index_dims_, index.data());
    SetCommonTensor(axis_, axis_dims_, axis.data());
  }
};

template <class T = float, class R = int32_t, class A = int32_t>
void TestGather(const std::vector<int64_t>& x_dims,
                const std::vector<int64_t>& index_dims,
                const std::vector<int64_t>& axis_dims,
                Place place,
                float abs_error = 1e-5,
                const std::string& alias = "def") {
  std::unique_ptr<arena::TestCase> tester(new GatherComputeTest<T, R, A>(
      place, alias, DDim(x_dims), DDim(index_dims), DDim(axis_dims)));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(Gather, precision) {
  float abs_error = 1e-5;
  Place place;
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
#elif defined(NNADAPTER_WITH_INTEL_OPENVINO)
  abs_error = 1e-5;
#elif defined(NNADAPTER_WITH_CAMBRICON_MLU)
  abs_error = 1e-3;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-3;
#else
  return;
#endif
#elif defined(LITE_WITH_NPU)
  place = TARGET(kNPU);
  abs_error = 1e-2;  // use fp16 in npu
  // TODO(zhupengyang): enable later
  return;
#elif defined(LITE_WITH_ARM)
  place = TARGET(kHost);
#elif defined(LITE_WITH_X86)
  place = TARGET(kX86);
#else
  return;
#endif

  for (auto x_dims : std::vector<std::vector<int64_t>>{
           {5, 7, 10, 12}, {8, 12, 16}, {12, 17}}) {
    for (auto index_dims : std::vector<std::vector<int64_t>>{{3}, {7}, {10}}) {
      for (auto axis_dims : std::vector<std::vector<int64_t>>{{1}, {0}}) {
#if (defined(LITE_WITH_NPU) || defined(LITE_WITH_NNADAPTER))
        axis_dims = {{0}};
        TestGather<float, int32_t, int32_t>(
            x_dims, index_dims, axis_dims, place, abs_error, "def");
#else
        TestGather<float, int64_t, int64_t>(
            x_dims, index_dims, axis_dims, place, abs_error, "int64int64");
        TestGather<float, int32_t, int32_t>(
            x_dims, index_dims, axis_dims, place, abs_error, "int32int32");
        TestGather<float, int32_t, int64_t>(
            x_dims, index_dims, axis_dims, place, abs_error, "int32int64");
        TestGather<float, int64_t, int32_t>(
            x_dims, index_dims, axis_dims, place, abs_error, "int64int32");
        TestGather<int64_t, int64_t, int64_t>(
            x_dims, index_dims, axis_dims, place, abs_error, "int64int64");
        TestGather<int64_t, int32_t, int32_t>(
            x_dims, index_dims, axis_dims, place, abs_error, "int32int32");
        TestGather<int64_t, int32_t, int64_t>(
            x_dims, index_dims, axis_dims, place, abs_error, "int32int64");
        TestGather<int64_t, int64_t, int32_t>(
            x_dims, index_dims, axis_dims, place, abs_error, "int64int32");
#endif
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle
