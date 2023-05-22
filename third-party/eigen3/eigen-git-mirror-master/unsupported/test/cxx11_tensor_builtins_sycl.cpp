// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016
// Mehdi Goli    Codeplay Software Ltd.
// Ralph Potter  Codeplay Software Ltd.
// Luke Iwanski  Codeplay Software Ltd.
// Contact: <eigen@codeplay.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int64_t
#define EIGEN_USE_SYCL

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;

namespace std {
template <typename T> T rsqrt(T x) { return 1 / std::sqrt(x); }
template <typename T> T square(T x) { return x * x; }
template <typename T> T cube(T x) { return x * x * x; }
template <typename T> T inverse(T x) { return 1 / x; }
}

#define TEST_UNARY_BUILTINS_FOR_SCALAR(FUNC, SCALAR, OPERATOR, Layout)         \
  {                                                                            \
    /* out OPERATOR in.FUNC() */                                               \
    Tensor<SCALAR, 3, Layout, int64_t> in(tensorRange);                        \
    Tensor<SCALAR, 3, Layout, int64_t> out(tensorRange);                       \
    in = in.random() + static_cast<SCALAR>(0.01);                              \
    out = out.random() + static_cast<SCALAR>(0.01);                            \
    Tensor<SCALAR, 3, Layout, int64_t> reference(out);                         \
    SCALAR *gpu_data = static_cast<SCALAR *>(                                  \
        sycl_device.allocate(in.size() * sizeof(SCALAR)));                     \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu(gpu_data, tensorRange);          \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
    sycl_device.memcpyHostToDevice(gpu_data, in.data(),                        \
                                   (in.size()) * sizeof(SCALAR));              \
    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
                                   (out.size()) * sizeof(SCALAR));             \
    gpu_out.device(sycl_device) OPERATOR gpu.FUNC();                           \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int64_t i = 0; i < out.size(); ++i) {                                 \
      SCALAR ver = reference(i);                                               \
      ver OPERATOR std::FUNC(in(i));                                           \
      VERIFY_IS_APPROX(out(i), ver);                                           \
    }                                                                          \
    sycl_device.deallocate(gpu_data);                                          \
    sycl_device.deallocate(gpu_data_out);                                      \
  }                                                                            \
  {                                                                            \
    /* out OPERATOR out.FUNC() */                                              \
    Tensor<SCALAR, 3, Layout, int64_t> out(tensorRange);                       \
    out = out.random() + static_cast<SCALAR>(0.01);                            \
    Tensor<SCALAR, 3, Layout, int64_t> reference(out);                         \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
    sycl_device.memcpyHostToDevice(gpu_data_out, out.data(),                   \
                                   (out.size()) * sizeof(SCALAR));             \
    gpu_out.device(sycl_device) OPERATOR gpu_out.FUNC();                       \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int64_t i = 0; i < out.size(); ++i) {                                 \
      SCALAR ver = reference(i);                                               \
      ver OPERATOR std::FUNC(reference(i));                                    \
      VERIFY_IS_APPROX(out(i), ver);                                           \
    }                                                                          \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_UNARY_BUILTINS_OPERATOR(SCALAR, OPERATOR , Layout)                \
  TEST_UNARY_BUILTINS_FOR_SCALAR(abs, SCALAR, OPERATOR , Layout)               \
  TEST_UNARY_BUILTINS_FOR_SCALAR(sqrt, SCALAR, OPERATOR , Layout)              \
  TEST_UNARY_BUILTINS_FOR_SCALAR(rsqrt, SCALAR, OPERATOR , Layout)             \
  TEST_UNARY_BUILTINS_FOR_SCALAR(square, SCALAR, OPERATOR , Layout)            \
  TEST_UNARY_BUILTINS_FOR_SCALAR(cube, SCALAR, OPERATOR , Layout)              \
  TEST_UNARY_BUILTINS_FOR_SCALAR(inverse, SCALAR, OPERATOR , Layout)           \
  TEST_UNARY_BUILTINS_FOR_SCALAR(tanh, SCALAR, OPERATOR , Layout)              \
  TEST_UNARY_BUILTINS_FOR_SCALAR(exp, SCALAR, OPERATOR , Layout)               \
  TEST_UNARY_BUILTINS_FOR_SCALAR(expm1, SCALAR, OPERATOR , Layout)             \
  TEST_UNARY_BUILTINS_FOR_SCALAR(log, SCALAR, OPERATOR , Layout)               \
  TEST_UNARY_BUILTINS_FOR_SCALAR(abs, SCALAR, OPERATOR , Layout)               \
  TEST_UNARY_BUILTINS_FOR_SCALAR(ceil, SCALAR, OPERATOR , Layout)              \
  TEST_UNARY_BUILTINS_FOR_SCALAR(floor, SCALAR, OPERATOR , Layout)             \
  TEST_UNARY_BUILTINS_FOR_SCALAR(round, SCALAR, OPERATOR , Layout)             \
  TEST_UNARY_BUILTINS_FOR_SCALAR(log1p, SCALAR, OPERATOR , Layout)

#define TEST_IS_THAT_RETURNS_BOOL(SCALAR, FUNC, Layout)                        \
  {                                                                            \
    /* out = in.FUNC() */                                                      \
    Tensor<SCALAR, 3, Layout, int64_t> in(tensorRange);                        \
    Tensor<bool, 3, Layout, int64_t> out(tensorRange);                         \
    in = in.random() + static_cast<SCALAR>(0.01);                              \
    SCALAR *gpu_data = static_cast<SCALAR *>(                                  \
        sycl_device.allocate(in.size() * sizeof(SCALAR)));                     \
    bool *gpu_data_out =                                                       \
        static_cast<bool *>(sycl_device.allocate(out.size() * sizeof(bool)));  \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu(gpu_data, tensorRange);          \
    TensorMap<Tensor<bool, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);    \
    sycl_device.memcpyHostToDevice(gpu_data, in.data(),                        \
                                   (in.size()) * sizeof(SCALAR));              \
    gpu_out.device(sycl_device) = gpu.FUNC();                                  \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(bool));               \
    for (int64_t i = 0; i < out.size(); ++i) {                                 \
      VERIFY_IS_EQUAL(out(i), std::FUNC(in(i)));                               \
    }                                                                          \
    sycl_device.deallocate(gpu_data);                                          \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_UNARY_BUILTINS(SCALAR, Layout)                                    \
  TEST_UNARY_BUILTINS_OPERATOR(SCALAR, +=, Layout)                             \
  TEST_UNARY_BUILTINS_OPERATOR(SCALAR, =, Layout)                              \
  TEST_IS_THAT_RETURNS_BOOL(SCALAR, isnan, Layout)                             \
  TEST_IS_THAT_RETURNS_BOOL(SCALAR, isfinite, Layout)                          \
  TEST_IS_THAT_RETURNS_BOOL(SCALAR, isinf, Layout)

static void test_builtin_unary_sycl(const Eigen::SyclDevice &sycl_device) {
  int64_t sizeDim1 = 10;
  int64_t sizeDim2 = 10;
  int64_t sizeDim3 = 10;
  array<int64_t, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};

  TEST_UNARY_BUILTINS(float, RowMajor)
  TEST_UNARY_BUILTINS(float, ColMajor)
}

namespace std {
template <typename T> T cwiseMax(T x, T y) { return std::max(x, y); }
template <typename T> T cwiseMin(T x, T y) { return std::min(x, y); }
}

#define TEST_BINARY_BUILTINS_FUNC(SCALAR, FUNC, Layout)                        \
  {                                                                            \
    /* out = in_1.FUNC(in_2) */                                                \
    Tensor<SCALAR, 3, Layout, int64_t> in_1(tensorRange);                      \
    Tensor<SCALAR, 3, Layout, int64_t> in_2(tensorRange);                      \
    Tensor<SCALAR, 3, Layout, int64_t> out(tensorRange);                       \
    in_1 = in_1.random() + static_cast<SCALAR>(0.01);                          \
    in_2 = in_2.random() + static_cast<SCALAR>(0.01);                          \
    Tensor<SCALAR, 3, Layout, int64_t> reference(out);                         \
    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
        sycl_device.allocate(in_1.size() * sizeof(SCALAR)));                   \
    SCALAR *gpu_data_2 = static_cast<SCALAR *>(                                \
        sycl_device.allocate(in_2.size() * sizeof(SCALAR)));                   \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_1(gpu_data_1, tensorRange);      \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_2(gpu_data_2, tensorRange);      \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
                                   (in_1.size()) * sizeof(SCALAR));            \
    sycl_device.memcpyHostToDevice(gpu_data_2, in_2.data(),                    \
                                   (in_2.size()) * sizeof(SCALAR));            \
    gpu_out.device(sycl_device) = gpu_1.FUNC(gpu_2);                           \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int64_t i = 0; i < out.size(); ++i) {                                 \
      SCALAR ver = reference(i);                                               \
      ver = std::FUNC(in_1(i), in_2(i));                                       \
      VERIFY_IS_APPROX(out(i), ver);                                           \
    }                                                                          \
    sycl_device.deallocate(gpu_data_1);                                        \
    sycl_device.deallocate(gpu_data_2);                                        \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_BINARY_BUILTINS_OPERATORS(SCALAR, OPERATOR, Layout)               \
  {                                                                            \
    /* out = in_1 OPERATOR in_2 */                                             \
    Tensor<SCALAR, 3, Layout, int64_t> in_1(tensorRange);                      \
    Tensor<SCALAR, 3, Layout, int64_t> in_2(tensorRange);                      \
    Tensor<SCALAR, 3, Layout, int64_t> out(tensorRange);                       \
    in_1 = in_1.random() + static_cast<SCALAR>(0.01);                          \
    in_2 = in_2.random() + static_cast<SCALAR>(0.01);                          \
    Tensor<SCALAR, 3, Layout, int64_t> reference(out);                         \
    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
        sycl_device.allocate(in_1.size() * sizeof(SCALAR)));                   \
    SCALAR *gpu_data_2 = static_cast<SCALAR *>(                                \
        sycl_device.allocate(in_2.size() * sizeof(SCALAR)));                   \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_1(gpu_data_1, tensorRange);      \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_2(gpu_data_2, tensorRange);      \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
                                   (in_1.size()) * sizeof(SCALAR));            \
    sycl_device.memcpyHostToDevice(gpu_data_2, in_2.data(),                    \
                                   (in_2.size()) * sizeof(SCALAR));            \
    gpu_out.device(sycl_device) = gpu_1 OPERATOR gpu_2;                        \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int64_t i = 0; i < out.size(); ++i) {                                 \
      VERIFY_IS_APPROX(out(i), in_1(i) OPERATOR in_2(i));                      \
    }                                                                          \
    sycl_device.deallocate(gpu_data_1);                                        \
    sycl_device.deallocate(gpu_data_2);                                        \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_BINARY_BUILTINS_OPERATORS_THAT_TAKES_SCALAR(SCALAR, OPERATOR, Layout)     \
  {                                                                            \
    /* out = in_1 OPERATOR 2 */                                                \
    Tensor<SCALAR, 3, Layout, int64_t> in_1(tensorRange);                      \
    Tensor<SCALAR, 3, Layout, int64_t> out(tensorRange);                       \
    in_1 = in_1.random() + static_cast<SCALAR>(0.01);                          \
    Tensor<SCALAR, 3, Layout, int64_t> reference(out);                         \
    SCALAR *gpu_data_1 = static_cast<SCALAR *>(                                \
        sycl_device.allocate(in_1.size() * sizeof(SCALAR)));                   \
    SCALAR *gpu_data_out = static_cast<SCALAR *>(                              \
        sycl_device.allocate(out.size() * sizeof(SCALAR)));                    \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_1(gpu_data_1, tensorRange);      \
    TensorMap<Tensor<SCALAR, 3, Layout, int64_t>> gpu_out(gpu_data_out, tensorRange);  \
    sycl_device.memcpyHostToDevice(gpu_data_1, in_1.data(),                    \
                                   (in_1.size()) * sizeof(SCALAR));            \
    gpu_out.device(sycl_device) = gpu_1 OPERATOR 2;                            \
    sycl_device.memcpyDeviceToHost(out.data(), gpu_data_out,                   \
                                   (out.size()) * sizeof(SCALAR));             \
    for (int64_t i = 0; i < out.size(); ++i) {                                 \
      VERIFY_IS_APPROX(out(i), in_1(i) OPERATOR 2);                            \
    }                                                                          \
    sycl_device.deallocate(gpu_data_1);                                        \
    sycl_device.deallocate(gpu_data_out);                                      \
  }

#define TEST_BINARY_BUILTINS(SCALAR, Layout)                                   \
  TEST_BINARY_BUILTINS_FUNC(SCALAR, cwiseMax , Layout)                         \
  TEST_BINARY_BUILTINS_FUNC(SCALAR, cwiseMin , Layout)                         \
  TEST_BINARY_BUILTINS_OPERATORS(SCALAR, + , Layout)                           \
  TEST_BINARY_BUILTINS_OPERATORS(SCALAR, - , Layout)                           \
  TEST_BINARY_BUILTINS_OPERATORS(SCALAR, * , Layout)                           \
  TEST_BINARY_BUILTINS_OPERATORS(SCALAR, / , Layout)

static void test_builtin_binary_sycl(const Eigen::SyclDevice &sycl_device) {
  int64_t sizeDim1 = 10;
  int64_t sizeDim2 = 10;
  int64_t sizeDim3 = 10;
  array<int64_t, 3> tensorRange = {{sizeDim1, sizeDim2, sizeDim3}};
  TEST_BINARY_BUILTINS(float, RowMajor)
  TEST_BINARY_BUILTINS_OPERATORS_THAT_TAKES_SCALAR(int, %, RowMajor)
  TEST_BINARY_BUILTINS(float, ColMajor)
  TEST_BINARY_BUILTINS_OPERATORS_THAT_TAKES_SCALAR(int, %, ColMajor)
}

EIGEN_DECLARE_TEST(cxx11_tensor_builtins_sycl) {
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    QueueInterface queueInterface(device);
    Eigen::SyclDevice sycl_device(&queueInterface);
    CALL_SUBTEST(test_builtin_unary_sycl(sycl_device));
    CALL_SUBTEST(test_builtin_binary_sycl(sycl_device));
  }
}
