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
#include <cmath>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"
#include "lite/tests/utils/fill_data.h"

namespace paddle {
namespace lite {

template <class Tx, class Ty>
void SequenceMask(const Tx* x, Ty* y, const int x_size, const int max_len) {
  memset(y, 0, sizeof(Ty) * x_size * max_len);
  for (int i = 0; i < x_size; i++) {
    int step = static_cast<int>(std::ceil(static_cast<float>(x[i])));
    for (int j = 0; j < step; j++) {
      y[j] = static_cast<Ty>(1);
    }
    y += max_len;
  }
}

template <class T>
class SequenceMaskTester : public arena::TestCase {
 protected:
  std::string x_ = "x";
  std::string max_len_tensor_;
  std::string y_ = "y";
  int max_len_{-1};
  int out_type_{5};
  DDim x_dims_{{2, 3, 4}};

 public:
  SequenceMaskTester(const Place& place,
                     const std::string& alias,
                     const int max_len = 5,
                     const int out_type = 5,
                     const bool use_max_len_tensor = false)
      : TestCase(place, alias), max_len_(max_len), out_type_(out_type) {
    if (use_max_len_tensor) {
      max_len_tensor_ = std::string("max_len_tensor");
    }
  }

  void RunBaseline(Scope* scope) override {
    auto* y = scope->NewTensor(y_);
    auto y_shape = x_dims_.Vectorize();
    y_shape.push_back(static_cast<int64_t>(max_len_));
    y->Resize(y_shape);

    auto* x = scope->FindTensor(x_);
    auto* x_data = x->template data<T>();
    int x_size = static_cast<int>(x->numel());

    switch (out_type_) {
      case 5: {
        SequenceMask(
            x_data, y->template mutable_data<float>(), x_size, max_len_);
        break;
      }
      case 2: {
        SequenceMask(x_data, y->template mutable_data<int>(), x_size, max_len_);
        break;
      }
      case 3: {
        SequenceMask(
            x_data, y->template mutable_data<int64_t>(), x_size, max_len_);
        break;
      }
      default:
        LOG(FATAL) << "unsupported out data type: " << out_type_;
        break;
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("sequence_mask");
    op_desc->SetInput("X", {x_});
    if (!max_len_tensor_.empty()) {
      op_desc->SetInput("MaxLenTensor", {max_len_tensor_});
      op_desc->SetAttr("maxlen", -1);
    } else {
      op_desc->SetAttr("maxlen", max_len_);
    }
    op_desc->SetOutput("Y", {y_});
    op_desc->SetAttr("out_dtype", out_type_);
  }

  void PrepareData() override {
    std::vector<T> x_data(x_dims_.production());
    fill_data_rand<T>(x_data.data(), 0, 4, x_dims_.production());
    SetCommonTensor(x_, x_dims_, x_data.data());

    if (!max_len_tensor_.empty()) {
      std::vector<int> max_len_tensor_data{max_len_};
      SetCommonTensor(max_len_tensor_, DDim{{1}}, max_len_tensor_data.data());
    }
  }
};

template <class T>
void TestSequenceMaskHelper(const Place place,
                            const float abs_error,
                            const int max_len = 5,
                            const int out_type = 5,
                            const bool use_max_len_tensor = false) {
  std::string alias;
  auto precision = lite_api::PrecisionTypeTrait<T>::Type();
  switch (precision) {
    case PRECISION(kFloat):
      alias = std::string("def");
      break;
    case PRECISION(kInt32):
      alias = std::string("int32");
      break;
    case PRECISION(kInt64):
      alias = std::string("int64");
      break;
    default:
      LOG(FATAL) << "unsupported input data type: "
                 << lite_api::PrecisionToStr(precision);
      break;
  }
  std::unique_ptr<arena::TestCase> tester(new SequenceMaskTester<T>(
      place, alias, max_len, out_type, use_max_len_tensor));
  arena::Arena arena(std::move(tester), place, abs_error);
  arena.TestPrecision();
}

template <class T>
void TestSequenceMask(const Place place, const float abs_error) {
  // test max_len
  for (int max_len : {6}) {
    TestSequenceMaskHelper<T>(place, abs_error, max_len);
  }
  // test out_type
  for (int out_type : {2, 3, 5}) {
    TestSequenceMaskHelper<T>(place, abs_error, 5, out_type);
  }
  // test max_len_tensor
  TestSequenceMaskHelper<T>(place, abs_error, 5, 5, true);
}

TEST(sequence_mask, precision) {
  Place place;
  float abs_error = 1e-5;
#if defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  place = TARGET(kHost);
#else
  return;
#endif

  TestSequenceMask<float>(place, abs_error);
  TestSequenceMask<int>(place, abs_error);
  TestSequenceMask<int64_t>(place, abs_error);
}

}  // namespace lite
}  // namespace paddle
