// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

template <int DataLayout>
static void test_simple_broadcasting()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  tensor.setRandom();
  array<ptrdiff_t, 4> broadcasts;
  broadcasts[0] = 1;
  broadcasts[1] = 1;
  broadcasts[2] = 1;
  broadcasts[3] = 1;

  Tensor<float, 4, DataLayout> no_broadcast;
  no_broadcast = tensor.broadcast(broadcasts);

  VERIFY_IS_EQUAL(no_broadcast.dimension(0), 2);
  VERIFY_IS_EQUAL(no_broadcast.dimension(1), 3);
  VERIFY_IS_EQUAL(no_broadcast.dimension(2), 5);
  VERIFY_IS_EQUAL(no_broadcast.dimension(3), 7);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i,j,k,l), no_broadcast(i,j,k,l));
        }
      }
    }
  }

  broadcasts[0] = 2;
  broadcasts[1] = 3;
  broadcasts[2] = 1;
  broadcasts[3] = 4;
  Tensor<float, 4, DataLayout> broadcast;
  broadcast = tensor.broadcast(broadcasts);

  VERIFY_IS_EQUAL(broadcast.dimension(0), 4);
  VERIFY_IS_EQUAL(broadcast.dimension(1), 9);
  VERIFY_IS_EQUAL(broadcast.dimension(2), 5);
  VERIFY_IS_EQUAL(broadcast.dimension(3), 28);

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 28; ++l) {
          VERIFY_IS_EQUAL(tensor(i%2,j%3,k%5,l%7), broadcast(i,j,k,l));
        }
      }
    }
  }
}


template <int DataLayout>
static void test_vectorized_broadcasting()
{
  Tensor<float, 3, DataLayout> tensor(8,3,5);
  tensor.setRandom();
  array<ptrdiff_t, 3> broadcasts;
  broadcasts[0] = 2;
  broadcasts[1] = 3;
  broadcasts[2] = 4;

  Tensor<float, 3, DataLayout> broadcast;
  broadcast = tensor.broadcast(broadcasts);

  VERIFY_IS_EQUAL(broadcast.dimension(0), 16);
  VERIFY_IS_EQUAL(broadcast.dimension(1), 9);
  VERIFY_IS_EQUAL(broadcast.dimension(2), 20);

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 20; ++k) {
        VERIFY_IS_EQUAL(tensor(i%8,j%3,k%5), broadcast(i,j,k));
      }
    }
  }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  tensor.resize(11,3,5);
#else
  array<Index, 3> new_dims;
  new_dims[0] = 11;
  new_dims[1] = 3;
  new_dims[2] = 5;
  tensor.resize(new_dims);
#endif

  tensor.setRandom();
  broadcast = tensor.broadcast(broadcasts);

  VERIFY_IS_EQUAL(broadcast.dimension(0), 22);
  VERIFY_IS_EQUAL(broadcast.dimension(1), 9);
  VERIFY_IS_EQUAL(broadcast.dimension(2), 20);

  for (int i = 0; i < 22; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 20; ++k) {
        VERIFY_IS_EQUAL(tensor(i%11,j%3,k%5), broadcast(i,j,k));
      }
    }
  }
}


template <int DataLayout>
static void test_static_broadcasting()
{
  Tensor<float, 3, DataLayout> tensor(8,3,5);
  tensor.setRandom();

#if defined(EIGEN_HAS_INDEX_LIST)
  Eigen::IndexList<Eigen::type2index<2>, Eigen::type2index<3>, Eigen::type2index<4>> broadcasts;
#else
  Eigen::array<int, 3> broadcasts;
  broadcasts[0] = 2;
  broadcasts[1] = 3;
  broadcasts[2] = 4;
#endif

  Tensor<float, 3, DataLayout> broadcast;
  broadcast = tensor.broadcast(broadcasts);

  VERIFY_IS_EQUAL(broadcast.dimension(0), 16);
  VERIFY_IS_EQUAL(broadcast.dimension(1), 9);
  VERIFY_IS_EQUAL(broadcast.dimension(2), 20);

  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 20; ++k) {
        VERIFY_IS_EQUAL(tensor(i%8,j%3,k%5), broadcast(i,j,k));
      }
    }
  }

#if EIGEN_HAS_VARIADIC_TEMPLATES
  tensor.resize(11,3,5);
#else
  array<Index, 3> new_dims;
  new_dims[0] = 11;
  new_dims[1] = 3;
  new_dims[2] = 5;
  tensor.resize(new_dims);
#endif

  tensor.setRandom();
  broadcast = tensor.broadcast(broadcasts);

  VERIFY_IS_EQUAL(broadcast.dimension(0), 22);
  VERIFY_IS_EQUAL(broadcast.dimension(1), 9);
  VERIFY_IS_EQUAL(broadcast.dimension(2), 20);

  for (int i = 0; i < 22; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 20; ++k) {
        VERIFY_IS_EQUAL(tensor(i%11,j%3,k%5), broadcast(i,j,k));
      }
    }
  }
}


template <int DataLayout>
static void test_fixed_size_broadcasting()
{
  // Need to add a [] operator to the Size class for this to work
#if 0
  Tensor<float, 1, DataLayout> t1(10);
  t1.setRandom();
  TensorFixedSize<float, Sizes<1>, DataLayout> t2;
  t2 = t2.constant(20.0f);

  Tensor<float, 1, DataLayout> t3 = t1 + t2.broadcast(Eigen::array<int, 1>{{10}});
  for (int i = 0; i < 10; ++i) {
    VERIFY_IS_APPROX(t3(i), t1(i) + t2(0));
  }

  TensorMap<TensorFixedSize<float, Sizes<1>, DataLayout> > t4(t2.data(), {{1}});
  Tensor<float, 1, DataLayout> t5 = t1 + t4.broadcast(Eigen::array<int, 1>{{10}});
  for (int i = 0; i < 10; ++i) {
    VERIFY_IS_APPROX(t5(i), t1(i) + t2(0));
  }
#endif
}

template <int DataLayout>
static void test_simple_broadcasting_one_by_n()
{
  Tensor<float, 4, DataLayout> tensor(1,13,5,7);
  tensor.setRandom();
  array<ptrdiff_t, 4> broadcasts;
  broadcasts[0] = 9;
  broadcasts[1] = 1;
  broadcasts[2] = 1;
  broadcasts[3] = 1;
  Tensor<float, 4, DataLayout> broadcast;
  broadcast = tensor.broadcast(broadcasts);

  VERIFY_IS_EQUAL(broadcast.dimension(0), 9);
  VERIFY_IS_EQUAL(broadcast.dimension(1), 13);
  VERIFY_IS_EQUAL(broadcast.dimension(2), 5);
  VERIFY_IS_EQUAL(broadcast.dimension(3), 7);

  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 13; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 7; ++l) {
          VERIFY_IS_EQUAL(tensor(i%1,j%13,k%5,l%7), broadcast(i,j,k,l));
        }
      }
    }
  }
}

template <int DataLayout>
static void test_simple_broadcasting_n_by_one()
{
  Tensor<float, 4, DataLayout> tensor(7,3,5,1);
  tensor.setRandom();
  array<ptrdiff_t, 4> broadcasts;
  broadcasts[0] = 1;
  broadcasts[1] = 1;
  broadcasts[2] = 1;
  broadcasts[3] = 19;
  Tensor<float, 4, DataLayout> broadcast;
  broadcast = tensor.broadcast(broadcasts);

  VERIFY_IS_EQUAL(broadcast.dimension(0), 7);
  VERIFY_IS_EQUAL(broadcast.dimension(1), 3);
  VERIFY_IS_EQUAL(broadcast.dimension(2), 5);
  VERIFY_IS_EQUAL(broadcast.dimension(3), 19);

  for (int i = 0; i < 7; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        for (int l = 0; l < 19; ++l) {
          VERIFY_IS_EQUAL(tensor(i%7,j%3,k%5,l%1), broadcast(i,j,k,l));
        }
      }
    }
  }
}

template <int DataLayout>
static void test_simple_broadcasting_one_by_n_by_one_1d()
{
  Tensor<float, 3, DataLayout> tensor(1,7,1);
  tensor.setRandom();
  array<ptrdiff_t, 3> broadcasts;
  broadcasts[0] = 5;
  broadcasts[1] = 1;
  broadcasts[2] = 13;
  Tensor<float, 3, DataLayout> broadcasted;
  broadcasted = tensor.broadcast(broadcasts);

  VERIFY_IS_EQUAL(broadcasted.dimension(0), 5);
  VERIFY_IS_EQUAL(broadcasted.dimension(1), 7);
  VERIFY_IS_EQUAL(broadcasted.dimension(2), 13);

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 13; ++k) {
        VERIFY_IS_EQUAL(tensor(0,j%7,0), broadcasted(i,j,k));
      }
    }
  }
}

template <int DataLayout>
static void test_simple_broadcasting_one_by_n_by_one_2d()
{
  Tensor<float, 4, DataLayout> tensor(1,7,13,1);
  tensor.setRandom();
  array<ptrdiff_t, 4> broadcasts;
  broadcasts[0] = 5;
  broadcasts[1] = 1;
  broadcasts[2] = 1;
  broadcasts[3] = 19;
  Tensor<float, 4, DataLayout> broadcast;
  broadcast = tensor.broadcast(broadcasts);

  VERIFY_IS_EQUAL(broadcast.dimension(0), 5);
  VERIFY_IS_EQUAL(broadcast.dimension(1), 7);
  VERIFY_IS_EQUAL(broadcast.dimension(2), 13);
  VERIFY_IS_EQUAL(broadcast.dimension(3), 19);

  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 7; ++j) {
      for (int k = 0; k < 13; ++k) {
        for (int l = 0; l < 19; ++l) {
          VERIFY_IS_EQUAL(tensor(0,j%7,k%13,0), broadcast(i,j,k,l));
        }
      }
    }
  }
}

EIGEN_DECLARE_TEST(cxx11_tensor_broadcasting)
{
  CALL_SUBTEST(test_simple_broadcasting<ColMajor>());
  CALL_SUBTEST(test_simple_broadcasting<RowMajor>());
  CALL_SUBTEST(test_vectorized_broadcasting<ColMajor>());
  CALL_SUBTEST(test_vectorized_broadcasting<RowMajor>());
  CALL_SUBTEST(test_static_broadcasting<ColMajor>());
  CALL_SUBTEST(test_static_broadcasting<RowMajor>());
  CALL_SUBTEST(test_fixed_size_broadcasting<ColMajor>());
  CALL_SUBTEST(test_fixed_size_broadcasting<RowMajor>());
  CALL_SUBTEST(test_simple_broadcasting_one_by_n<RowMajor>());
  CALL_SUBTEST(test_simple_broadcasting_n_by_one<RowMajor>());
  CALL_SUBTEST(test_simple_broadcasting_one_by_n<ColMajor>());
  CALL_SUBTEST(test_simple_broadcasting_n_by_one<ColMajor>());
  CALL_SUBTEST(test_simple_broadcasting_one_by_n_by_one_1d<ColMajor>());
  CALL_SUBTEST(test_simple_broadcasting_one_by_n_by_one_2d<ColMajor>());
  CALL_SUBTEST(test_simple_broadcasting_one_by_n_by_one_1d<RowMajor>());
  CALL_SUBTEST(test_simple_broadcasting_one_by_n_by_one_2d<RowMajor>());
}
