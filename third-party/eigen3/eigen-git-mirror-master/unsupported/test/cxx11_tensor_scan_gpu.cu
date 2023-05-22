// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2016 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int
#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

#include <Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

using Eigen::Tensor;
typedef Tensor<float, 1>::DimensionPair DimPair;

template<int DataLayout>
void test_gpu_cumsum(int m_size, int k_size, int n_size)
{
  std::cout << "Testing for (" << m_size << "," << k_size << "," << n_size << ")" << std::endl;
  Tensor<float, 3, DataLayout> t_input(m_size, k_size, n_size);
  Tensor<float, 3, DataLayout> t_result(m_size, k_size, n_size);
  Tensor<float, 3, DataLayout> t_result_gpu(m_size, k_size, n_size);

  t_input.setRandom();

  std::size_t t_input_bytes = t_input.size()  * sizeof(float);
  std::size_t t_result_bytes = t_result.size() * sizeof(float);

  float* d_t_input;
  float* d_t_result;

  gpuMalloc((void**)(&d_t_input), t_input_bytes);
  gpuMalloc((void**)(&d_t_result), t_result_bytes);

  gpuMemcpy(d_t_input, t_input.data(), t_input_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<float, 3, DataLayout> >
      gpu_t_input(d_t_input, Eigen::array<int, 3>(m_size, k_size, n_size));
  Eigen::TensorMap<Eigen::Tensor<float, 3, DataLayout> >
      gpu_t_result(d_t_result, Eigen::array<int, 3>(m_size, k_size, n_size));

  gpu_t_result.device(gpu_device) = gpu_t_input.cumsum(1);
  t_result = t_input.cumsum(1);

  gpuMemcpy(t_result_gpu.data(), d_t_result, t_result_bytes, gpuMemcpyDeviceToHost);
  for (DenseIndex i = 0; i < t_result.size(); i++) {
    if (fabs(t_result(i) - t_result_gpu(i)) < 1e-4f) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), 1e-4f)) {
      continue;
    }
    std::cout << "mismatch detected at index " << i << ": " << t_result(i)
              << " vs " <<  t_result_gpu(i) << std::endl;
    assert(false);
  }

  gpuFree((void*)d_t_input);
  gpuFree((void*)d_t_result);
}


EIGEN_DECLARE_TEST(cxx11_tensor_scan_gpu)
{
  CALL_SUBTEST_1(test_gpu_cumsum<ColMajor>(128, 128, 128));
  CALL_SUBTEST_2(test_gpu_cumsum<RowMajor>(128, 128, 128));
}
