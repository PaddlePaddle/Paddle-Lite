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

#include <cmath>
#include <string>
#include <vector>

#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"
#include "lite/core/op_registry.h"

template <class T>
bool is_fp_close(T v1, T v2, T rel_tol = 1e-4, T abs_tol = 1e-5) {
  bool abs_chk = std::abs(v1 - v2) < abs_tol;
  bool rel_chk =
      (std::abs(v1 - v2) / std::min(std::abs(v1), std::abs(v2))) < rel_tol;
  return abs_chk || rel_chk;
}

template <class T>
T* AtLogicInd(T* data,
              const std::vector<int>& dim,
              const std::vector<int>& logic_index) {
  assert(dim.size() == logic_index.size());

  int offset = 0;
  int stride = 1;
  for (int i = dim.size() - 1; i >= 0; --i) {
    int ind = logic_index[i];
    if (dim[i] == 1) {
      ind = 0;
    }
    assert(ind < dim[i]);
    offset += ind * stride;
    stride *= dim[i];
  }
  return data + offset;
}

std::vector<int> GenLogicIndex(int logic_offset, const std::vector<int>& dim) {
  std::vector<int> strides(dim.size(), 1);
  for (int i = dim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dim[i + 1];
  }
  std::vector<int> ret(dim.size(), 0);
  for (int i = 0; i < dim.size(); ++i) {
    ret[i] = logic_offset / strides[i];
    logic_offset %= strides[i];
  }
  return ret;
}
template <class T>
void BroadcastCPURef(const T* x,
                     const T* y,
                     T* z,
                     const std::vector<int>& x_dim,
                     const std::vector<int>& y_dim,
                     const std::vector<int>& z_dim,
                     const std::function<T(T, T)> op) {
  int N = 1;
  for (int i = 0; i < z_dim.size(); ++i) {
    N *= z_dim[i];
  }
  for (int i = 0; i < N; ++i) {
    auto logic_index = GenLogicIndex(i, z_dim);
    const T* x_d = AtLogicInd(x, x_dim, logic_index);
    const T* y_d = AtLogicInd(y, y_dim, logic_index);
    T* z_d = AtLogicInd(z, z_dim, logic_index);
    *z_d = op(*x_d, *y_d);
  }
}

int randint(int beg, int end) {
  static unsigned int seed = 0;
  int rd = rand_r(&seed);
  int range = end - beg + 1;
  rd = rd % range;
  return rd + beg;
}

bool randbool() { return randint(0, 1000000) < 50000; }

namespace paddle {
namespace lite {
template <class T>
class ElementwiseComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "x";
  std::string y_ = "y";
  std::string out_ = "out";
  std::string elt_type_ = "";  // add, sub, mul, div, max
  paddle::lite::DDim x_dims_;
  paddle::lite::DDim y_dims_;
  paddle::lite::DDim out_dims_;
  std::vector<int> x_dim_full_;
  std::vector<int> y_dim_full_;
  int axis_ = 1;
  std::string act_type_ = "";
  const std::function<T(T, T)> op_;

 public:
  ElementwiseComputeTester(const Place& place,
                           const std::string& alias,
                           const std::string& elt_type,
                           const std::vector<int64_t>& x_shape,
                           const std::vector<int64_t>& y_shape,
                           const std::vector<int64_t>& out_shape,
                           const std::vector<int>& x_dim_full,
                           const std::vector<int>& y_dim_full,
                           int axis,
                           std::string act_type,
                           const std::function<T(T, T)> op)
      : TestCase(place, alias),
        elt_type_(elt_type),
        x_dims_(DDim(x_shape)),
        y_dims_(DDim(y_shape)),
        out_dims_(DDim(out_shape)),
        x_dim_full_(x_dim_full),
        y_dim_full_(y_dim_full),
        axis_(axis),
        act_type_(act_type),
        op_(op) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto y = scope->FindTensor(y_);
    auto out = scope->NewTensor(out_);
    out->Resize(out_dims_);
    auto i64_out_dim = out->dims().Vectorize();
    std::vector<int32_t> out_dim(i64_out_dim.begin(), i64_out_dim.end());
    BroadcastCPURef<T>(x->data<T>(),
                       y->data<T>(),
                       out->mutable_data<T>(),
                       x_dim_full_,
                       y_dim_full_,
                       out_dim,
                       op_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) override {
    std::string op_type = "elementwise_" + elt_type_;
    if (!act_type_.empty()) {
      op_type = "fusion_" + op_type + "_activation";
    }
    op_desc->SetType(op_type);
    op_desc->SetInput("X", {x_});
    op_desc->SetInput("Y", {y_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr("axis", axis_);
    op_desc->SetAttr("act_type", act_type_);
  }

  void PrepareData() override {
    std::vector<T> dx(x_dims_.production());
    for (size_t i = 0; i < dx.size(); i++) {
      dx[i] = i;
    }
    SetCommonTensor(x_, x_dims_, dx.data());

    std::vector<T> dy(y_dims_.production());
    for (size_t i = 0; i < dy.size(); i++) {
      dy[i] = 2 * i + 1;
    }
    SetCommonTensor(y_, y_dims_, dy.data());
  }
};

template <class T>
bool RunOnRandomArgs(const Place& place,
                     const std::string& alias,
                     const std::string& elt_type,
                     const std::string& act_type,
                     const std::function<T(T, T)> op,
                     double abs_error = 1e-3) {
  const int MAX_DIM_SIZE = 8;
  const int MAX_SHAPE_VALUE = 10;
  // gen out_dim
  int dim_size = randint(2, MAX_DIM_SIZE);
  std::vector<int> out_dim(dim_size, 0);
  for (int i = 0; i < dim_size; ++i) {
    out_dim[i] = randint(2, MAX_SHAPE_VALUE);
  }

  // gen a random broad-cast able x_dim and y_dim
  std::vector<int> x_dim_full = out_dim;
  std::vector<int> y_dim_full = out_dim;

  std::vector<int> x_dim_cut;
  std::vector<int> y_dim_cut;

  int axis = -1;
  bool cut_dimension = randbool();
  if (cut_dimension) {
    // generate x_dim_cut and y_dim_cut by remove dimension
    bool use_axis = randbool();
    if (use_axis) {
      x_dim_cut = x_dim_full;
      // we will cut y only, and set tail of y to be 1
      axis = randint(0, dim_size - 1);
      int tail1_num = randint(0, dim_size - axis);
      for (int i = 0; i < axis; ++i) {
        y_dim_full[i] = 1;
      }
      for (int i = axis; i < (dim_size - tail1_num); ++i) {
        y_dim_cut.push_back(y_dim_full[i]);
      }
      for (int i = 0; i < tail1_num; ++i) {
        y_dim_cut.push_back(1);
      }
      for (int i = dim_size - tail1_num; i < dim_size; ++i) {
        y_dim_full[i] = 1;
      }
    } else {
      // we will cut x or y
      if (randbool()) {
        y_dim_cut = y_dim_full;
        int cut_x_num = randint(0, dim_size) * randbool();
        for (int i = 0; i < cut_x_num; ++i) {
          x_dim_full[i] = 1;
        }
        for (int i = cut_x_num; i < dim_size; ++i) {
          x_dim_cut.push_back(x_dim_full[i]);
        }
      } else {
        x_dim_cut = x_dim_full;
        int cut_y_num = randint(0, dim_size) * randbool();
        for (int i = 0; i < cut_y_num; ++i) {
          y_dim_full[i] = 1;
        }
        for (int i = cut_y_num; i < dim_size; ++i) {
          y_dim_cut.push_back(y_dim_full[i]);
        }
      }
    }
  } else {
    // generate x_dim_cut and y_dim_cut by random
    // random assign 1 to some dim
    for (int i = 0; i < dim_size; ++i) {
      if (randbool() && y_dim_full[i] != 1) {
        x_dim_full[i] = 1;
      }
      if (randbool() && x_dim_full[i] != 1) {
        y_dim_full[i] = 1;
      }
    }
    // just remove 1 at high dimesion
    int ind = 0;
    while (x_dim_full[ind] == 1) {
      ++ind;
    }
    for (int i = ind; i < dim_size; ++i) {
      x_dim_cut.push_back(x_dim_full[i]);
    }
    ind = 0;
    while (y_dim_full[ind] == 1) {
      ++ind;
    }
    for (int i = ind; i < dim_size; ++i) {
      y_dim_cut.push_back(y_dim_full[i]);
    }
  }

  // run on device
  auto x_dim = std::vector<int64_t>(x_dim_cut.begin(), x_dim_cut.end());
  auto y_dim = std::vector<int64_t>(y_dim_cut.begin(), y_dim_cut.end());
  auto z_dim = std::vector<int64_t>(out_dim.begin(), out_dim.end());
  std::unique_ptr<arena::TestCase> tester(
      new ElementwiseComputeTester<T>(place,
                                      alias,
                                      elt_type,
                                      x_dim,
                                      y_dim,
                                      z_dim,
                                      x_dim_full,
                                      y_dim_full,
                                      axis,
                                      act_type,
                                      op));
  arena::Arena arena(std::move(tester), place, abs_error);
  return arena.TestPrecision();
}

}  // namespace lite
}  // namespace paddle

#ifdef LITE_WITH_ARM

TEST(elementwise_broadcast, compute_fp32) {
  const int TEST_RETEAT_NUM = 5;
  for (int repeat_count = 0; repeat_count < TEST_RETEAT_NUM; ++repeat_count) {
    EXPECT_TRUE(paddle::lite::RunOnRandomArgs<float>(
        TARGET(kARM), "def", "add", "", [](float l, float r) {
          return l + r;
        }));
    EXPECT_TRUE(paddle::lite::RunOnRandomArgs<float>(
        TARGET(kARM), "def", "sub", "", [](float l, float r) {
          return l - r;
        }));
    EXPECT_TRUE(paddle::lite::RunOnRandomArgs<float>(
        TARGET(kARM), "def", "mul", "", [](float l, float r) {
          return l * r;
        }));
    EXPECT_TRUE(paddle::lite::RunOnRandomArgs<float>(
        TARGET(kARM), "def", "div", "", [](float l, float r) {
          return l / r;
        }));
  }
}

TEST(elementwise_broadcast, compute_i32) {
  const int TEST_RETEAT_NUM = 5;
  for (int repeat_count = 0; repeat_count < TEST_RETEAT_NUM; ++repeat_count) {
    EXPECT_TRUE(paddle::lite::RunOnRandomArgs<int32_t>(
        paddle::lite::Place(TARGET(kARM), PRECISION(kInt32)),
        "def",
        "add",
        "",
        [](int32_t l, int32_t r) { return l + r; }));
    EXPECT_TRUE(paddle::lite::RunOnRandomArgs<int32_t>(
        paddle::lite::Place(TARGET(kARM), PRECISION(kInt32)),
        "def",
        "sub",
        "",
        [](int32_t l, int32_t r) { return l - r; }));
    EXPECT_TRUE(paddle::lite::RunOnRandomArgs<int32_t>(
        paddle::lite::Place(TARGET(kARM), PRECISION(kInt32)),
        "def",
        "mul",
        "",
        [](int32_t l, int32_t r) { return l * r; }));
    EXPECT_TRUE(paddle::lite::RunOnRandomArgs<int32_t>(
        paddle::lite::Place(TARGET(kARM), PRECISION(kInt32)),
        "def",
        "div",
        "",
        [](int32_t l, int32_t r) { return l / r; }));
  }
}

#endif  // LITE_WITH_ARM
