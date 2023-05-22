// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <limits>
#include <numeric>
#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <int DataLayout>
static void test_trivial_reductions() {
  {
    Tensor<float, 0, DataLayout> tensor;
    tensor.setRandom();
    array<ptrdiff_t, 0> reduction_axis;

    Tensor<float, 0, DataLayout> result = tensor.sum(reduction_axis);
    VERIFY_IS_EQUAL(result(), tensor());
  }

  {
    Tensor<float, 1, DataLayout> tensor(7);
    tensor.setRandom();
    array<ptrdiff_t, 0> reduction_axis;

    Tensor<float, 1, DataLayout> result = tensor.sum(reduction_axis);
    VERIFY_IS_EQUAL(result.dimension(0), 7);
    for (int i = 0; i < 7; ++i) {
      VERIFY_IS_EQUAL(result(i), tensor(i));
    }
  }

  {
    Tensor<float, 2, DataLayout> tensor(2, 3);
    tensor.setRandom();
    array<ptrdiff_t, 0> reduction_axis;

    Tensor<float, 2, DataLayout> result = tensor.sum(reduction_axis);
    VERIFY_IS_EQUAL(result.dimension(0), 2);
    VERIFY_IS_EQUAL(result.dimension(1), 3);
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        VERIFY_IS_EQUAL(result(i, j), tensor(i, j));
      }
    }
  }
}

template <typename Scalar,int DataLayout>
static void test_simple_reductions() {
  Tensor<Scalar, 4, DataLayout> tensor(2, 3, 5, 7);
  tensor.setRandom();
  // Add a little offset so that the product reductions won't be close to zero.
  tensor += tensor.constant(Scalar(0.5f));
  array<ptrdiff_t, 2> reduction_axis2;
  reduction_axis2[0] = 1;
  reduction_axis2[1] = 3;

  Tensor<Scalar, 2, DataLayout> result = tensor.sum(reduction_axis2);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_EQUAL(result.dimension(1), 5);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 5; ++j) {
      Scalar sum = Scalar(0.0f);
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 7; ++l) {
          sum += tensor(i, k, j, l);
        }
      }
      VERIFY_IS_APPROX(result(i, j), sum);
    }
  }

  {
    Tensor<Scalar, 0, DataLayout> sum1 = tensor.sum();
    VERIFY_IS_EQUAL(sum1.rank(), 0);

    array<ptrdiff_t, 4> reduction_axis4;
    reduction_axis4[0] = 0;
    reduction_axis4[1] = 1;
    reduction_axis4[2] = 2;
    reduction_axis4[3] = 3;
    Tensor<Scalar, 0, DataLayout> sum2 = tensor.sum(reduction_axis4);
    VERIFY_IS_EQUAL(sum2.rank(), 0);

    VERIFY_IS_APPROX(sum1(), sum2());
  }

  reduction_axis2[0] = 0;
  reduction_axis2[1] = 2;
  result = tensor.prod(reduction_axis2);
  VERIFY_IS_EQUAL(result.dimension(0), 3);
  VERIFY_IS_EQUAL(result.dimension(1), 7);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      Scalar prod = Scalar(1.0f);
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 5; ++l) {
          prod *= tensor(k, i, l, j);
        }
      }
      VERIFY_IS_APPROX(result(i, j), prod);
    }
  }

  {
    Tensor<Scalar, 0, DataLayout> prod1 = tensor.prod();
    VERIFY_IS_EQUAL(prod1.rank(), 0);

    array<ptrdiff_t, 4> reduction_axis4;
    reduction_axis4[0] = 0;
    reduction_axis4[1] = 1;
    reduction_axis4[2] = 2;
    reduction_axis4[3] = 3;
    Tensor<Scalar, 0, DataLayout> prod2 = tensor.prod(reduction_axis4);
    VERIFY_IS_EQUAL(prod2.rank(), 0);

    VERIFY_IS_APPROX(prod1(), prod2());
  }

  reduction_axis2[0] = 0;
  reduction_axis2[1] = 2;
  result = tensor.maximum(reduction_axis2);
  VERIFY_IS_EQUAL(result.dimension(0), 3);
  VERIFY_IS_EQUAL(result.dimension(1), 7);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 7; ++j) {
      Scalar max_val = std::numeric_limits<Scalar>::lowest();
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 5; ++l) {
          max_val = (std::max)(max_val, tensor(k, i, l, j));
        }
      }
      VERIFY_IS_APPROX(result(i, j), max_val);
    }
  }

  {
    Tensor<Scalar, 0, DataLayout> max1 = tensor.maximum();
    VERIFY_IS_EQUAL(max1.rank(), 0);

    array<ptrdiff_t, 4> reduction_axis4;
    reduction_axis4[0] = 0;
    reduction_axis4[1] = 1;
    reduction_axis4[2] = 2;
    reduction_axis4[3] = 3;
    Tensor<Scalar, 0, DataLayout> max2 = tensor.maximum(reduction_axis4);
    VERIFY_IS_EQUAL(max2.rank(), 0);

    VERIFY_IS_APPROX(max1(), max2());
  }

  reduction_axis2[0] = 0;
  reduction_axis2[1] = 1;
  result = tensor.minimum(reduction_axis2);
  VERIFY_IS_EQUAL(result.dimension(0), 5);
  VERIFY_IS_EQUAL(result.dimension(1), 7);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 7; ++j) {
      Scalar min_val = (std::numeric_limits<Scalar>::max)();
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 3; ++l) {
          min_val = (std::min)(min_val, tensor(k, l, i, j));
        }
      }
      VERIFY_IS_APPROX(result(i, j), min_val);
    }
  }

  {
    Tensor<Scalar, 0, DataLayout> min1 = tensor.minimum();
    VERIFY_IS_EQUAL(min1.rank(), 0);

    array<ptrdiff_t, 4> reduction_axis4;
    reduction_axis4[0] = 0;
    reduction_axis4[1] = 1;
    reduction_axis4[2] = 2;
    reduction_axis4[3] = 3;
    Tensor<Scalar, 0, DataLayout> min2 = tensor.minimum(reduction_axis4);
    VERIFY_IS_EQUAL(min2.rank(), 0);

    VERIFY_IS_APPROX(min1(), min2());
  }

  reduction_axis2[0] = 0;
  reduction_axis2[1] = 1;
  result = tensor.mean(reduction_axis2);
  VERIFY_IS_EQUAL(result.dimension(0), 5);
  VERIFY_IS_EQUAL(result.dimension(1), 7);
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 7; ++j) {
      Scalar sum = Scalar(0.0f);
      int count = 0;
      for (int k = 0; k < 2; ++k) {
        for (int l = 0; l < 3; ++l) {
          sum += tensor(k, l, i, j);
          ++count;
        }
      }
      VERIFY_IS_APPROX(result(i, j), sum / count);
    }
  }

  {
    Tensor<Scalar, 0, DataLayout> mean1 = tensor.mean();
    VERIFY_IS_EQUAL(mean1.rank(), 0);

    array<ptrdiff_t, 4> reduction_axis4;
    reduction_axis4[0] = 0;
    reduction_axis4[1] = 1;
    reduction_axis4[2] = 2;
    reduction_axis4[3] = 3;
    Tensor<Scalar, 0, DataLayout> mean2 = tensor.mean(reduction_axis4);
    VERIFY_IS_EQUAL(mean2.rank(), 0);

    VERIFY_IS_APPROX(mean1(), mean2());
  }

  {
    Tensor<int, 1> ints(10);
    std::iota(ints.data(), ints.data() + ints.dimension(0), 0);

    TensorFixedSize<bool, Sizes<> > all_;
    all_ = ints.all();
    VERIFY(!all_());
    all_ = (ints >= ints.constant(0)).all();
    VERIFY(all_());

    TensorFixedSize<bool, Sizes<> > any;
    any = (ints > ints.constant(10)).any();
    VERIFY(!any());
    any = (ints < ints.constant(1)).any();
    VERIFY(any());
  }
}


template <int DataLayout>
static void test_reductions_in_expr() {
  Tensor<float, 4, DataLayout> tensor(2, 3, 5, 7);
  tensor.setRandom();
  array<ptrdiff_t, 2> reduction_axis2;
  reduction_axis2[0] = 1;
  reduction_axis2[1] = 3;

  Tensor<float, 2, DataLayout> result(2, 5);
  result = result.constant(1.0f) - tensor.sum(reduction_axis2);
  VERIFY_IS_EQUAL(result.dimension(0), 2);
  VERIFY_IS_EQUAL(result.dimension(1), 5);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 5; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 7; ++l) {
          sum += tensor(i, k, j, l);
        }
      }
      VERIFY_IS_APPROX(result(i, j), 1.0f - sum);
    }
  }
}


template <int DataLayout>
static void test_full_reductions() {
  Tensor<float, 2, DataLayout> tensor(2, 3);
  tensor.setRandom();
  array<ptrdiff_t, 2> reduction_axis;
  reduction_axis[0] = 0;
  reduction_axis[1] = 1;

  Tensor<float, 0, DataLayout> result = tensor.sum(reduction_axis);
  VERIFY_IS_EQUAL(result.rank(), 0);

  float sum = 0.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      sum += tensor(i, j);
    }
  }
  VERIFY_IS_APPROX(result(0), sum);

  result = tensor.square().sum(reduction_axis).sqrt();
  VERIFY_IS_EQUAL(result.rank(), 0);

  sum = 0.0f;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      sum += tensor(i, j) * tensor(i, j);
    }
  }
  VERIFY_IS_APPROX(result(), sqrtf(sum));
}

struct UserReducer {
  static const bool PacketAccess = false;
  UserReducer(float offset) : offset_(offset) {}
  void reduce(const float val, float* accum) { *accum += val * val; }
  float initialize() const { return 0; }
  float finalize(const float accum) const { return 1.0f / (accum + offset_); }

 private:
  const float offset_;
};

template <int DataLayout>
static void test_user_defined_reductions() {
  Tensor<float, 2, DataLayout> tensor(5, 7);
  tensor.setRandom();
  array<ptrdiff_t, 1> reduction_axis;
  reduction_axis[0] = 1;

  UserReducer reducer(10.0f);
  Tensor<float, 1, DataLayout> result = tensor.reduce(reduction_axis, reducer);
  VERIFY_IS_EQUAL(result.dimension(0), 5);
  for (int i = 0; i < 5; ++i) {
    float expected = 10.0f;
    for (int j = 0; j < 7; ++j) {
      expected += tensor(i, j) * tensor(i, j);
    }
    expected = 1.0f / expected;
    VERIFY_IS_APPROX(result(i), expected);
  }
}

template <int DataLayout>
static void test_tensor_maps() {
  int inputs[2 * 3 * 5 * 7];
  TensorMap<Tensor<int, 4, DataLayout> > tensor_map(inputs, 2, 3, 5, 7);
  TensorMap<Tensor<const int, 4, DataLayout> > tensor_map_const(inputs, 2, 3, 5,
                                                                7);
  const TensorMap<Tensor<const int, 4, DataLayout> > tensor_map_const_const(
      inputs, 2, 3, 5, 7);

  tensor_map.setRandom();
  array<ptrdiff_t, 2> reduction_axis;
  reduction_axis[0] = 1;
  reduction_axis[1] = 3;

  Tensor<int, 2, DataLayout> result = tensor_map.sum(reduction_axis);
  Tensor<int, 2, DataLayout> result2 = tensor_map_const.sum(reduction_axis);
  Tensor<int, 2, DataLayout> result3 =
      tensor_map_const_const.sum(reduction_axis);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 5; ++j) {
      int sum = 0;
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 7; ++l) {
          sum += tensor_map(i, k, j, l);
        }
      }
      VERIFY_IS_EQUAL(result(i, j), sum);
      VERIFY_IS_EQUAL(result2(i, j), sum);
      VERIFY_IS_EQUAL(result3(i, j), sum);
    }
  }
}

template <int DataLayout>
static void test_static_dims() {
  Tensor<float, 4, DataLayout> in(72, 53, 97, 113);
  Tensor<float, 2, DataLayout> out(72, 97);
  in.setRandom();

#if !EIGEN_HAS_CONSTEXPR
  array<int, 2> reduction_axis;
  reduction_axis[0] = 1;
  reduction_axis[1] = 3;
#else
  Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<3> > reduction_axis;
#endif

  out = in.maximum(reduction_axis);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 97; ++j) {
      float expected = -1e10f;
      for (int k = 0; k < 53; ++k) {
        for (int l = 0; l < 113; ++l) {
          expected = (std::max)(expected, in(i, k, j, l));
        }
      }
      VERIFY_IS_EQUAL(out(i, j), expected);
    }
  }
}

template <int DataLayout>
static void test_innermost_last_dims() {
  Tensor<float, 4, DataLayout> in(72, 53, 97, 113);
  Tensor<float, 2, DataLayout> out(97, 113);
  in.setRandom();

// Reduce on the innermost dimensions.
#if !EIGEN_HAS_CONSTEXPR
  array<int, 2> reduction_axis;
  reduction_axis[0] = 0;
  reduction_axis[1] = 1;
#else
  // This triggers the use of packets for ColMajor.
  Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<1> > reduction_axis;
#endif

  out = in.maximum(reduction_axis);

  for (int i = 0; i < 97; ++i) {
    for (int j = 0; j < 113; ++j) {
      float expected = -1e10f;
      for (int k = 0; k < 53; ++k) {
        for (int l = 0; l < 72; ++l) {
          expected = (std::max)(expected, in(l, k, i, j));
        }
      }
      VERIFY_IS_EQUAL(out(i, j), expected);
    }
  }
}

template <int DataLayout>
static void test_innermost_first_dims() {
  Tensor<float, 4, DataLayout> in(72, 53, 97, 113);
  Tensor<float, 2, DataLayout> out(72, 53);
  in.setRandom();

// Reduce on the innermost dimensions.
#if !EIGEN_HAS_CONSTEXPR
  array<int, 2> reduction_axis;
  reduction_axis[0] = 2;
  reduction_axis[1] = 3;
#else
  // This triggers the use of packets for RowMajor.
  Eigen::IndexList<Eigen::type2index<2>, Eigen::type2index<3>> reduction_axis;
#endif

  out = in.maximum(reduction_axis);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 53; ++j) {
      float expected = -1e10f;
      for (int k = 0; k < 97; ++k) {
        for (int l = 0; l < 113; ++l) {
          expected = (std::max)(expected, in(i, j, k, l));
        }
      }
      VERIFY_IS_EQUAL(out(i, j), expected);
    }
  }
}

template <int DataLayout>
static void test_reduce_middle_dims() {
  Tensor<float, 4, DataLayout> in(72, 53, 97, 113);
  Tensor<float, 2, DataLayout> out(72, 53);
  in.setRandom();

// Reduce on the innermost dimensions.
#if !EIGEN_HAS_CONSTEXPR
  array<int, 2> reduction_axis;
  reduction_axis[0] = 1;
  reduction_axis[1] = 2;
#else
  // This triggers the use of packets for RowMajor.
  Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2>> reduction_axis;
#endif

  out = in.maximum(reduction_axis);

  for (int i = 0; i < 72; ++i) {
    for (int j = 0; j < 113; ++j) {
      float expected = -1e10f;
      for (int k = 0; k < 53; ++k) {
        for (int l = 0; l < 97; ++l) {
          expected = (std::max)(expected, in(i, k, l, j));
        }
      }
      VERIFY_IS_EQUAL(out(i, j), expected);
    }
  }
}

static void test_sum_accuracy() {
  Tensor<float, 3> tensor(101, 101, 101);
  for (float prescribed_mean : {1.0f, 10.0f, 100.0f, 1000.0f, 10000.0f}) {
    tensor.setRandom();
    tensor += tensor.constant(prescribed_mean);

    Tensor<float, 0> sum = tensor.sum();
    double expected_sum = 0.0;
    for (int i = 0; i < 101; ++i) {
      for (int j = 0; j < 101; ++j) {
        for (int k = 0; k < 101; ++k) {
          expected_sum += static_cast<double>(tensor(i, j, k));
        }
      }
    }
    VERIFY_IS_APPROX(sum(), static_cast<float>(expected_sum));
  }
}

EIGEN_DECLARE_TEST(cxx11_tensor_reduction) {
  CALL_SUBTEST(test_trivial_reductions<ColMajor>());
  CALL_SUBTEST(test_trivial_reductions<RowMajor>());
  CALL_SUBTEST(( test_simple_reductions<float,ColMajor>() ));
  CALL_SUBTEST(( test_simple_reductions<float,RowMajor>() ));
  CALL_SUBTEST(( test_simple_reductions<Eigen::half,ColMajor>() ));
  CALL_SUBTEST(test_reductions_in_expr<ColMajor>());
  CALL_SUBTEST(test_reductions_in_expr<RowMajor>());
  CALL_SUBTEST(test_full_reductions<ColMajor>());
  CALL_SUBTEST(test_full_reductions<RowMajor>());
  CALL_SUBTEST(test_user_defined_reductions<ColMajor>());
  CALL_SUBTEST(test_user_defined_reductions<RowMajor>());
  CALL_SUBTEST(test_tensor_maps<ColMajor>());
  CALL_SUBTEST(test_tensor_maps<RowMajor>());
  CALL_SUBTEST(test_static_dims<ColMajor>());
  CALL_SUBTEST(test_static_dims<RowMajor>());
  CALL_SUBTEST(test_innermost_last_dims<ColMajor>());
  CALL_SUBTEST(test_innermost_last_dims<RowMajor>());
  CALL_SUBTEST(test_innermost_first_dims<ColMajor>());
  CALL_SUBTEST(test_innermost_first_dims<RowMajor>());
  CALL_SUBTEST(test_reduce_middle_dims<ColMajor>());
  CALL_SUBTEST(test_reduce_middle_dims<RowMajor>());
  CALL_SUBTEST(test_sum_accuracy());
}
