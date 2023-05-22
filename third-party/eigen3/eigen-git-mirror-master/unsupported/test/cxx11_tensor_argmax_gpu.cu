// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


#define EIGEN_TEST_NO_LONGDOUBLE

#define EIGEN_USE_GPU

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>

using Eigen::Tensor;

template <int Layout>
void test_gpu_simple_argmax()
{
  Tensor<double, 3, Layout> in(Eigen::array<DenseIndex, 3>(72,53,97));
  Tensor<DenseIndex, 1, Layout> out_max(Eigen::array<DenseIndex, 1>(1));
  Tensor<DenseIndex, 1, Layout> out_min(Eigen::array<DenseIndex, 1>(1));
  in.setRandom();
  in *= in.constant(100.0);
  in(0, 0, 0) = -1000.0;
  in(71, 52, 96) = 1000.0;

  std::size_t in_bytes = in.size() * sizeof(double);
  std::size_t out_bytes = out_max.size() * sizeof(DenseIndex);

  double* d_in;
  DenseIndex* d_out_max;
  DenseIndex* d_out_min;
  gpuMalloc((void**)(&d_in), in_bytes);
  gpuMalloc((void**)(&d_out_max), out_bytes);
  gpuMalloc((void**)(&d_out_min), out_bytes);

  gpuMemcpy(d_in, in.data(), in_bytes, gpuMemcpyHostToDevice);

  Eigen::GpuStreamDevice stream;
  Eigen::GpuDevice gpu_device(&stream);

  Eigen::TensorMap<Eigen::Tensor<double, 3, Layout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 3>(72,53,97));
  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_max(d_out_max, Eigen::array<DenseIndex, 1>(1));
  Eigen::TensorMap<Eigen::Tensor<DenseIndex, 1, Layout>, Aligned > gpu_out_min(d_out_min, Eigen::array<DenseIndex, 1>(1));

  gpu_out_max.device(gpu_device) = gpu_in.argmax();
  gpu_out_min.device(gpu_device) = gpu_in.argmin();

  assert(gpuMemcpyAsync(out_max.data(), d_out_max, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuMemcpyAsync(out_min.data(), d_out_min, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
  assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

  VERIFY_IS_EQUAL(out_max(Eigen::array<DenseIndex, 1>(0)), 72*53*97 - 1);
  VERIFY_IS_EQUAL(out_min(Eigen::array<DenseIndex, 1>(0)), 0);

  gpuFree(d_in);
  gpuFree(d_out_max);
  gpuFree(d_out_min);
}

template <int DataLayout>
void test_gpu_argmax_dim()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  std::vector<int> dims;
  dims.push_back(2); dims.push_back(3); dims.push_back(5); dims.push_back(7);

  for (int dim = 0; dim < 4; ++dim) {
    tensor.setRandom();
    tensor = (tensor + tensor.constant(0.5)).log();

    array<DenseIndex, 3> out_shape;
    for (int d = 0; d < 3; ++d) out_shape[d] = (d < dim) ? dims[d] : dims[d+1];

    Tensor<DenseIndex, 3, DataLayout> tensor_arg(out_shape);

    array<DenseIndex, 4> ix;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i; ix[1] = j; ix[2] = k; ix[3] = l;
            if (ix[dim] != 0) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 0, k, l) = 10.0
            tensor(ix) = 10.0;
          }
        }
      }
    }

    std::size_t in_bytes = tensor.size() * sizeof(float);
    std::size_t out_bytes = tensor_arg.size() * sizeof(DenseIndex);

    float* d_in;
    DenseIndex* d_out;
    gpuMalloc((void**)(&d_in), in_bytes);
    gpuMalloc((void**)(&d_out), out_bytes);

    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);

    Eigen::GpuStreamDevice stream;
    Eigen::GpuDevice gpu_device(&stream);

    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);

    gpu_out.device(gpu_device) = gpu_in.argmax(dim);

    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

    VERIFY_IS_EQUAL(tensor_arg.size(),
                    size_t(2*3*5*7 / tensor.dimension(dim)));

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect max to be in the first index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], 0);
    }

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i; ix[1] = j; ix[2] = k; ix[3] = l;
            if (ix[dim] != tensor.dimension(dim) - 1) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 2, k, l) = 20.0
            tensor(ix) = 20.0;
          }
        }
      }
    }

    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);

    gpu_out.device(gpu_device) = gpu_in.argmax(dim);

    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect max to be in the last index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], tensor.dimension(dim) - 1);
    }

    gpuFree(d_in);
    gpuFree(d_out);
  }
}

template <int DataLayout>
void test_gpu_argmin_dim()
{
  Tensor<float, 4, DataLayout> tensor(2,3,5,7);
  std::vector<int> dims;
  dims.push_back(2); dims.push_back(3); dims.push_back(5); dims.push_back(7);

  for (int dim = 0; dim < 4; ++dim) {
    tensor.setRandom();
    tensor = (tensor + tensor.constant(0.5)).log();

    array<DenseIndex, 3> out_shape;
    for (int d = 0; d < 3; ++d) out_shape[d] = (d < dim) ? dims[d] : dims[d+1];

    Tensor<DenseIndex, 3, DataLayout> tensor_arg(out_shape);

    array<DenseIndex, 4> ix;
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i; ix[1] = j; ix[2] = k; ix[3] = l;
            if (ix[dim] != 0) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 0, k, l) = 10.0
            tensor(ix) = -10.0;
          }
        }
      }
    }

    std::size_t in_bytes = tensor.size() * sizeof(float);
    std::size_t out_bytes = tensor_arg.size() * sizeof(DenseIndex);

    float* d_in;
    DenseIndex* d_out;
    gpuMalloc((void**)(&d_in), in_bytes);
    gpuMalloc((void**)(&d_out), out_bytes);

    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);

    Eigen::GpuStreamDevice stream;
    Eigen::GpuDevice gpu_device(&stream);

    Eigen::TensorMap<Eigen::Tensor<float, 4, DataLayout>, Aligned > gpu_in(d_in, Eigen::array<DenseIndex, 4>(2, 3, 5, 7));
    Eigen::TensorMap<Eigen::Tensor<DenseIndex, 3, DataLayout>, Aligned > gpu_out(d_out, out_shape);

    gpu_out.device(gpu_device) = gpu_in.argmin(dim);

    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

    VERIFY_IS_EQUAL(tensor_arg.size(),
                    2*3*5*7 / tensor.dimension(dim));

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect min to be in the first index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], 0);
    }

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        for (int k = 0; k < 5; ++k) {
          for (int l = 0; l < 7; ++l) {
            ix[0] = i; ix[1] = j; ix[2] = k; ix[3] = l;
            if (ix[dim] != tensor.dimension(dim) - 1) continue;
            // suppose dim == 1, then for all i, k, l, set tensor(i, 2, k, l) = 20.0
            tensor(ix) = -20.0;
          }
        }
      }
    }

    gpuMemcpy(d_in, tensor.data(), in_bytes, gpuMemcpyHostToDevice);

    gpu_out.device(gpu_device) = gpu_in.argmin(dim);

    assert(gpuMemcpyAsync(tensor_arg.data(), d_out, out_bytes, gpuMemcpyDeviceToHost, gpu_device.stream()) == gpuSuccess);
    assert(gpuStreamSynchronize(gpu_device.stream()) == gpuSuccess);

    for (DenseIndex n = 0; n < tensor_arg.size(); ++n) {
      // Expect max to be in the last index of the reduced dimension
      VERIFY_IS_EQUAL(tensor_arg.data()[n], tensor.dimension(dim) - 1);
    }

    gpuFree(d_in);
    gpuFree(d_out);
  }
}

EIGEN_DECLARE_TEST(cxx11_tensor_argmax_gpu)
{
  CALL_SUBTEST_1(test_gpu_simple_argmax<RowMajor>());
  CALL_SUBTEST_1(test_gpu_simple_argmax<ColMajor>());
  CALL_SUBTEST_2(test_gpu_argmax_dim<RowMajor>());
  CALL_SUBTEST_2(test_gpu_argmax_dim<ColMajor>());
  CALL_SUBTEST_3(test_gpu_argmin_dim<RowMajor>());
  CALL_SUBTEST_3(test_gpu_argmin_dim<ColMajor>());
}
