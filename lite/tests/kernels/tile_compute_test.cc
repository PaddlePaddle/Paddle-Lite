// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

template <class T>
class TileComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  DDim x_dims_;
  std::string out_ = "Out";
  std::string repeat_times_tensor_;
  std::vector<std::string> repeat_times_tensor_list_;
  std::vector<int> repeat_times_;

 public:
  TileComputeTester(const Place& place,
                    const std::string& alias,
                    const DDim& x_dims,
                    const std::vector<int>& repeat_times,
                    const bool use_repeat_times_tensor = false,
                    const bool use_repeat_times_tensor_list = false)
      : TestCase(place, alias), x_dims_(x_dims), repeat_times_(repeat_times) {
    if (use_repeat_times_tensor) {
      repeat_times_tensor_ = "RepeatTimes";
    }
    if (use_repeat_times_tensor_list) {
      for (size_t i = 0; i < repeat_times.size(); i++) {
        repeat_times_tensor_list_.push_back("repeat_times_" +
                                            std::to_string(i));
      }
    }
  }

  void RunBaseline(Scope* scope) override {
    std::vector<int64_t> vec_in_dims = x_dims_.Vectorize();
    std::vector<int> repeat_times = repeat_times_;
    // broadcast for vec_in_dims.size() equal to repeat_times.size()
    if (repeat_times.size() < vec_in_dims.size()) {
      int diff = vec_in_dims.size() - repeat_times.size();
      repeat_times.insert(repeat_times.begin(), diff, 1);
    } else {
      int diff = repeat_times.size() - vec_in_dims.size();
      vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    }

    DDim new_in_dims{vec_in_dims};
    DDim out_dims(new_in_dims);
    std::vector<int> bcast_dims(vec_in_dims.size() + 1);
    std::vector<int> in_stride(vec_in_dims.size() + 1);

    in_stride[0] = 1;
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      bcast_dims[i] = repeat_times[i];
      out_dims[i] *= repeat_times[i];
      if (i > 0) {
        in_stride[i + 1] = in_stride[i] / new_in_dims[i - 1];
      } else {
        in_stride[i + 1] = new_in_dims.production();
      }
    }
    bcast_dims[repeat_times.size()] = 1;

    const auto* in = scope->FindTensor(x_);
    auto* out = scope->NewTensor(out_);
    out->Resize(out_dims);

    Tensor tmp_src_tensor;
    Tensor tmp_dst_tensor;
    auto in_data = in->template data<T>();
    tmp_src_tensor.Resize(out_dims);
    tmp_dst_tensor.Resize(out_dims);
    auto tmp_src = tmp_src_tensor.mutable_data<T>();
    auto tmp_dst = tmp_dst_tensor.mutable_data<T>();
    for (int i = 0; i < x_dims_.production(); i++) {
      tmp_src[i] = in_data[i];
      tmp_dst[i] = in_data[i];
    }

    int right = 1;
    for (int i = bcast_dims.size() - 1; i >= 0; i--) {
      right *= bcast_dims[i];
      if (bcast_dims[i] > 1) {
        int num = in_stride[1] / in_stride[i + 1];
        int dst_stride = in_stride[i + 1] * right;
        for (int m = 0; m < num; m++) {
          for (int j = 0; j < bcast_dims[i]; j++) {
            std::memcpy(
                tmp_dst + j * dst_stride / bcast_dims[i] + m * dst_stride,
                tmp_src + m * dst_stride / bcast_dims[i],
                dst_stride / bcast_dims[i] * sizeof(T));
          }
        }
        tmp_src_tensor.CopyDataFrom(tmp_dst_tensor);
      }
    }
    out->CopyDataFrom(tmp_dst_tensor);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("tile");
    op_desc->SetInput("X", {x_});
    if (!repeat_times_tensor_.empty()) {
      op_desc->SetInput("RepeatTimes", {repeat_times_tensor_});
    }
    if (!repeat_times_tensor_list_.empty()) {
      op_desc->SetInput("repeat_times_tensor", repeat_times_tensor_list_);
    }
    op_desc->SetOutput("Out", {out_});
    if (repeat_times_tensor_.empty() && repeat_times_tensor_list_.empty()) {
      op_desc->SetAttr("repeat_times", repeat_times_);
    } else {
      op_desc->SetAttr("repeat_times", std::vector<int>{});
    }
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    fill_data_rand(x_data.data(),
                   static_cast<T>(-5),
                   static_cast<T>(5),
                   x_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data());

    if (!repeat_times_tensor_.empty()) {
      SetCommonTensor(repeat_times_tensor_,
                      DDim({static_cast<int64_t>(repeat_times_.size())}),
                      repeat_times_.data(),
                      {},
                      true);
    }

    if (!repeat_times_tensor_list_.empty()) {
      for (size_t i = 0; i < repeat_times_.size(); i++) {
        SetCommonTensor(repeat_times_tensor_list_[i],
                        DDim({1}),
                        repeat_times_.data() + i,
                        {},
                        true);
      }
    }
  }
};

template <class T>
void TestTile(Place place,
              std::string alias,
              float abs_error,
              const std::vector<int64_t>& x_shape = {2, 1, 4},
              const std::vector<int>& repeat_times = {2, 3, 4},
              const bool use_repeat_times_tensor = false,
              const bool use_repeat_times_tensor_list = false) {
  std::unique_ptr<arena::TestCase> tester(
      new TileComputeTester<T>(place,
                               alias,
                               DDim(x_shape),
                               repeat_times,
                               use_repeat_times_tensor,
                               use_repeat_times_tensor_list));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

TEST(tile, precision) {
  Place place;
  float abs_error = 1e-5;
  std::string alias{"def_float"};
#if defined(LITE_WITH_NNADAPTER)
  place = TARGET(kNNAdapter);
#if defined(NNADAPTER_WITH_HUAWEI_ASCEND_NPU)
  abs_error = 1e-2;
  alias = "def";
  TestTile<float>(place, alias, abs_error);
  TestTile<float>(place, alias, abs_error, {1, 1, 1}, {2, 3, 4});
  TestTile<float>(place, alias, abs_error, {2, 1, 4}, {2, 3, 4}, true);
  return;
#elif defined(NNADAPTER_WITH_HUAWEI_KIRIN_NPU)
  abs_error = 1e-2;
  alias = "def";
#elif defined(NNADAPTER_WITH_QUALCOMM_QNN)
  abs_error = 1e-2;
  alias = "def";
#else
  return;
#endif
#elif defined(LITE_WITH_XPU)
  place = TARGET(kXPU);
  alias = "def";
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestTile<float>(place, alias, abs_error);
  TestTile<float>(place, alias, abs_error, {1, 1, 1}, {2, 3, 4});
  TestTile<float>(place, alias, abs_error, {2, 1, 4}, {2, 3, 4}, true);
}

}  // namespace lite
}  // namespace paddle
