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
#ifdef defined(LITE_WITH_ARM)

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/arm/elementwise_compute.h"

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

using paddle::lite::PrecisionType;
template <template <class, PrecisionType> class ElementWiseComputeTemplate,
          typename T,
          PrecisionType PType>
void do_broadcast_compute(std::function<T(T, T)> op) {
  ElementWiseComputeTemplate<T, PType> elementwise_op;

  const int MAX_DIM_SIZE = 8;
  const int MAX_SHAPE_VALUE = 10;
  // gen out_dim
  int dim_size = randint(2, MAX_DIM_SIZE);
  std::vector<int> out_dim(dim_size, 0);
  for (int i = 0; i < dim_size; ++i) {
    out_dim[i] = randint(1, MAX_SHAPE_VALUE);
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

  // run on kernel
  paddle::lite::Tensor x, y, output;
  auto x_dim = paddle::lite::DDim(
      std::vector<int64_t>(x_dim_cut.begin(), x_dim_cut.end()));
  auto y_dim = paddle::lite::DDim(
      std::vector<int64_t>(y_dim_cut.begin(), y_dim_cut.end()));
  x.Resize(x_dim);
  y.Resize(y_dim);
  output.Resize(
      paddle::lite::DDim(std::vector<int64_t>(out_dim.begin(), out_dim.end())));
  T* x_data = x.mutable_data<T>();
  T* y_data = y.mutable_data<T>();
  T* output_data = output.mutable_data<T>();

  static unsigned int rand_seed = 0;
  for (int i = 0; i < x_dim.production(); i++) {
    x_data[i] = 1.0 * rand_r(&rand_seed) * rand_r(&rand_seed) /
                (rand_r(&rand_seed) + 1);
  }
  for (int i = 0; i < y_dim.production(); i++) {
    y_data[i] = 1.0 * rand_r(&rand_seed) * rand_r(&rand_seed) /
                (rand_r(&rand_seed) + 1);
  }
  paddle::lite::operators::ElementwiseParam param;
  param.X = &x;
  param.Y = &y;
  param.axis = axis;
  param.Out = &output;
  elementwise_op.SetParam(param);
  elementwise_op.Run();

  // run on test
  std::vector<T> cpu_ref(output.dims().production(), -1);
  BroadcastCPURef<T>(
      x_data, y_data, cpu_ref.data(), x_dim_full, y_dim_full, out_dim, op);

  // cmp

  if (std::is_floating_point<T>::value) {
    for (int i = 0; i < output.dims().production(); i++) {
      ASSERT_EQ(is_fp_close(output_data[i], cpu_ref[i]), true)
          << "Value differ at index " << i;
    }
  } else {
    for (int i = 0; i < output.dims().production(); i++) {
      ASSERT_EQ(output_data[i], cpu_ref[i]) << "Value differ at index " << i;
    }
  }
}

using paddle::lite::kernels::arm::ElementwiseAddCompute;
using paddle::lite::kernels::arm::ElementwiseDivCompute;
using paddle::lite::kernels::arm::ElementwiseMulCompute;
using paddle::lite::kernels::arm::ElementwiseSubCompute;

TEST(elementwise_broadcast, compute_fp32) {
  const int TEST_RETEAT_NUM = 5;
  for (int repeat_count = 0; repeat_count < TEST_RETEAT_NUM; ++repeat_count) {
    do_broadcast_compute<ElementwiseAddCompute, float, PRECISION(kFloat)>(
        [](float l, float r) { return l + r; });
    do_broadcast_compute<ElementwiseSubCompute, float, PRECISION(kFloat)>(
        [](float l, float r) { return l - r; });
    do_broadcast_compute<ElementwiseMulCompute, float, PRECISION(kFloat)>(
        [](float l, float r) { return l * r; });
    do_broadcast_compute<ElementwiseDivCompute, float, PRECISION(kFloat)>(
        [](float l, float r) { return l / r; });
    if (::testing::Test::HasFailure()) {
      FAIL();
    }
  }
}

TEST(elementwise_broadcast, compute_i32) {
  const int TEST_RETEAT_NUM = 5;
  for (int repeat_count = 0; repeat_count < TEST_RETEAT_NUM; ++repeat_count) {
    do_broadcast_compute<ElementwiseAddCompute, int32_t, PRECISION(kInt32)>(
        [](int32_t l, int32_t r) { return l + r; });
    do_broadcast_compute<ElementwiseSubCompute, int32_t, PRECISION(kInt32)>(
        [](int32_t l, int32_t r) { return l - r; });
    do_broadcast_compute<ElementwiseMulCompute, int32_t, PRECISION(kInt32)>(
        [](int32_t l, int32_t r) { return l * r; });
    do_broadcast_compute<ElementwiseDivCompute, int32_t, PRECISION(kInt32)>(
        [](int32_t l, int32_t r) { return l / r; });
    if (::testing::Test::HasFailure()) {
      FAIL();
    }
  }
}

#endif  // LITE_WITH_ARM
