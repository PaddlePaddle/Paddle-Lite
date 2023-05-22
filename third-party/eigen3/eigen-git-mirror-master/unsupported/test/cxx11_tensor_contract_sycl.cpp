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

#include <iostream>
#include <chrono>
#include <ctime>

#include "main.h"
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::array;
using Eigen::SyclDevice;
using Eigen::Tensor;
using Eigen::TensorMap;
template<int DataLayout, typename DataType, typename IndexType, typename Device>
void static test_sycl_contraction(const Device& sycl_device, IndexType m_size, IndexType k_size, IndexType n_size)
{
  typedef typename Tensor<DataType, 1, DataLayout, IndexType>::DimensionPair DimPair;
  static const DataType error_threshold =1e-4f;
//  std::cout << "Testing for (" << m_size << "," << k_size << "," << n_size << ")" << std::endl;
  // with these dimensions, the output has 300 * 140 elements, which is
  // more than 30 * 1024, which is the number of threads in blocks on
  // a 15 SM GK110 GPU
  Tensor<DataType, 2, DataLayout, IndexType> t_left(m_size, k_size);
  Tensor<DataType, 2, DataLayout, IndexType> t_right(k_size, n_size);
  Tensor<DataType, 2, DataLayout, IndexType> t_result(m_size, n_size);
  Tensor<DataType, 2, DataLayout, IndexType> t_result_gpu(m_size, n_size);
//  Eigen::array<DimPair, 1> dims(DimPair(1, 0));
  Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};
  Eigen::array<IndexType, 2> left_dims = {{m_size, k_size}};
  Eigen::array<IndexType, 2> right_dims = {{k_size, n_size}};
  Eigen::array<IndexType, 2> result_dims = {{m_size, n_size}};

  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size()  * sizeof(DataType);
  std::size_t t_right_bytes = t_right.size() * sizeof(DataType);
  std::size_t t_result_bytes = t_result.size() * sizeof(DataType);

  DataType * d_t_left  = static_cast<DataType*>(sycl_device.allocate(t_left_bytes));
  DataType * d_t_right  = static_cast<DataType*>(sycl_device.allocate(t_right_bytes));
  DataType * d_t_result =  static_cast<DataType*>(sycl_device.allocate(t_result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_left(d_t_left, left_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_right(d_t_right, right_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_result(d_t_result, result_dims);

  sycl_device.memcpyHostToDevice(d_t_left, t_left.data(),t_left_bytes);
  sycl_device.memcpyHostToDevice(d_t_right, t_right.data(),t_right_bytes);

  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result, t_result_bytes);

  t_result = t_left.contract(t_right, dims);

  for (IndexType i = 0; i < t_result.size(); i++) {
    if (static_cast<DataType>(fabs(t_result(i) - t_result_gpu(i))) < error_threshold) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), error_threshold)) {
      continue;
    }
    std::cout << "mismatch detected at IndexType " << i << ": " << t_result(i)
              << " vs " <<  t_result_gpu(i) << std::endl;
    assert(false);
  }
  sycl_device.deallocate(d_t_left);
  sycl_device.deallocate(d_t_right);
  sycl_device.deallocate(d_t_result);
}

template<int DataLayout, typename DataType, typename IndexType, typename Device>
void test_TF(const Device& sycl_device)
{
  typedef typename Tensor<DataType, 1, DataLayout, IndexType>::DimensionPair DimPair;
  static const DataType error_threshold =1e-4f;
  Eigen::array<IndexType, 2> left_dims = {{2, 3}};
  Eigen::array<IndexType, 2> right_dims = {{3, 1}};
  Eigen::array<IndexType, 2> res_dims = {{2, 1}};
  Eigen::array<DimPair, 1> dims = {{DimPair(1, 0)}};


  Tensor<DataType, 2, DataLayout, IndexType> t_left(left_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_right(right_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_result_gpu(res_dims);
  Tensor<DataType, 2, DataLayout, IndexType> t_result(res_dims);

  t_left.data()[0] = 1.0f;
  t_left.data()[1] = 2.0f;
  t_left.data()[2] = 3.0f;
  t_left.data()[3] = 4.0f;
  t_left.data()[4] = 5.0f;
  t_left.data()[5] = 6.0f;

  t_right.data()[0] = -1.0f;
  t_right.data()[1] = 0.5f;
  t_right.data()[2] = 2.0f;

  std::size_t t_left_bytes = t_left.size()  * sizeof(DataType);
  std::size_t t_right_bytes = t_right.size() * sizeof(DataType);
  std::size_t t_result_bytes = t_result.size()*sizeof(DataType);


  DataType * d_t_left  = static_cast<DataType*>(sycl_device.allocate(t_left_bytes));
  DataType * d_t_right  = static_cast<DataType*>(sycl_device.allocate(t_right_bytes));
  DataType * d_t_result =  static_cast<DataType*>(sycl_device.allocate(t_result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_left(d_t_left, left_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_right(d_t_right, right_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_result(d_t_result, res_dims);

  sycl_device.memcpyHostToDevice(d_t_left, t_left.data(),t_left_bytes);
  sycl_device.memcpyHostToDevice(d_t_right, t_right.data(),t_right_bytes);

  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result, t_result_bytes);

  t_result = t_left.contract(t_right, dims);

  for (IndexType i = 0; i < t_result.size(); i++) {
    if (static_cast<DataType>(fabs(t_result(i) - t_result_gpu(i))) < error_threshold) {
      continue;
    }
    if (Eigen::internal::isApprox(t_result(i), t_result_gpu(i), error_threshold)) {
      continue;
    }
    std::cout << "mismatch detected at IndexType " << i << ": " << t_result(i)
              << " vs " <<  t_result_gpu(i) << std::endl;
    assert(false);
  }
  sycl_device.deallocate(d_t_left);
  sycl_device.deallocate(d_t_right);
  sycl_device.deallocate(d_t_result);


}

template<int DataLayout,  typename DataType, typename IndexType, typename Device>
void test_scalar(const Device& sycl_device, IndexType m_size, IndexType k_size, IndexType n_size)
{
  //std::cout << "Testing for (" << m_size << "," << k_size << "," << n_size << ")" << std::endl;
  // with these dimensions, the output has 300 * 140 elements, which is
  // more than 30 * 1024, which is the number of threads in blocks on
  // a 15 SM GK110 GPU
  typedef typename Tensor<DataType, 1, DataLayout, IndexType>::DimensionPair DimPair;
  static const DataType error_threshold =1e-4f;
  Tensor<DataType, 2, DataLayout, IndexType> t_left(m_size, k_size);
  Tensor<DataType, 2, DataLayout, IndexType> t_right(k_size, n_size);
  Tensor<DataType, 0, DataLayout, IndexType> t_result;
  Tensor<DataType, 0, DataLayout, IndexType> t_result_gpu;
  Eigen::array<DimPair, 2> dims = {{DimPair(0, 0), DimPair(1, 1)}};
  Eigen::array<IndexType, 2> left_dims = {{m_size, k_size}};
  Eigen::array<IndexType, 2> right_dims = {{k_size, n_size}};
  t_left.setRandom();
  t_right.setRandom();

  std::size_t t_left_bytes = t_left.size()  * sizeof(DataType);
  std::size_t t_right_bytes = t_right.size() * sizeof(DataType);
  std::size_t t_result_bytes = sizeof(DataType);


  DataType * d_t_left  = static_cast<DataType*>(sycl_device.allocate(t_left_bytes));
  DataType * d_t_right  = static_cast<DataType*>(sycl_device.allocate(t_right_bytes));
  DataType * d_t_result =  static_cast<DataType*>(sycl_device.allocate(t_result_bytes));

  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_left(d_t_left, left_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 2, DataLayout, IndexType> > gpu_t_right(d_t_right, right_dims);
  Eigen::TensorMap<Eigen::Tensor<DataType, 0, DataLayout, IndexType> > gpu_t_result(d_t_result);

  sycl_device.memcpyHostToDevice(d_t_left, t_left.data(),t_left_bytes);
  sycl_device.memcpyHostToDevice(d_t_right, t_right.data(),t_right_bytes);

  gpu_t_result.device(sycl_device) = gpu_t_left.contract(gpu_t_right, dims);
  sycl_device.memcpyDeviceToHost(t_result_gpu.data(), d_t_result, t_result_bytes);

  t_result = t_left.contract(t_right, dims);

  if (static_cast<DataType>(fabs(t_result() - t_result_gpu())) > error_threshold &&
      !Eigen::internal::isApprox(t_result(), t_result_gpu(), error_threshold)) {
    std::cout << "mismatch detected: " << t_result()
              << " vs " <<  t_result_gpu() << std::endl;
    assert(false);
  }

  sycl_device.deallocate(d_t_left);
  sycl_device.deallocate(d_t_right);
  sycl_device.deallocate(d_t_result);
}


template<int DataLayout, typename DataType, typename IndexType, typename Device>
void test_sycl_contraction_m(const Device& sycl_device) {
  for (IndexType k = 32; k < 256; k++) {
    test_sycl_contraction<DataLayout, DataType, IndexType>(sycl_device, k, 128, 128);
  }
}

template<int DataLayout,  typename DataType, typename IndexType, typename Device>
void test_sycl_contraction_k(const Device& sycl_device) {
  for (IndexType k = 32; k < 256; k++) {
    test_sycl_contraction<DataLayout, DataType, IndexType>(sycl_device, 128, k, 128);
  }
}

template<int DataLayout,  typename DataType, typename IndexType, typename Device>
void test_sycl_contraction_n(const Device& sycl_device) {
  for (IndexType k = 32; k < 256; k++) {
    test_sycl_contraction<DataLayout, DataType, IndexType>(sycl_device, 128, 128, k);
  }
}


template<int DataLayout,  typename DataType, typename IndexType, typename Device>
void test_sycl_contraction_sizes(const Device& sycl_device) {
  IndexType m_sizes[] = { 31,  39,   63,   64,   65,
                   127, 129,  255,  257 , 511,
                   512, 513, 1023, 1024, 1025};

  IndexType n_sizes[] = { 31,  39,   63,   64,   65,
                   127, 129,  255,  257,  511,
                   512, 513, 1023, 1024, 1025};

  IndexType k_sizes[] = {  31,   39,  63,  64,   65,
                     95,   96, 127, 129,  255,
                    257,  511, 512, 513, 1023,
                   1024, 1025};

  for (IndexType i = 0; i < 15; i++) {
    for (IndexType j = 0; j < 15; j++) {
      for (IndexType k = 0; k < 17; k++) {
        test_sycl_contraction<DataLayout, DataType,IndexType>(sycl_device, m_sizes[i], n_sizes[j], k_sizes[k]);
      }
    }
  }
}

template <typename Dev_selector> void tensorContractionPerDevice(Dev_selector& s){
  QueueInterface queueInterface(s);
  auto sycl_device=Eigen::SyclDevice(&queueInterface);
  test_sycl_contraction<ColMajor, float,int64_t>(sycl_device, 32, 32, 32);
  test_sycl_contraction<RowMajor,float,int64_t>(sycl_device, 32, 32, 32);
  test_scalar<ColMajor,float,int64_t>(sycl_device, 32, 32, 32);
  test_scalar<RowMajor,float,int64_t>(sycl_device, 32, 32, 32);
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  test_sycl_contraction<ColMajor,float,int64_t>(sycl_device, 128, 128, 128);
  test_sycl_contraction<RowMajor,float,int64_t>(sycl_device, 128, 128, 128);
  test_scalar<ColMajor,float,int64_t>(sycl_device, 128, 128, 128);
  test_scalar<RowMajor,float,int64_t>(sycl_device, 128, 128, 128);
  test_sycl_contraction_m<ColMajor, float, int64_t>(sycl_device);
  test_sycl_contraction_m<RowMajor, float, int64_t>(sycl_device);
  test_sycl_contraction_n<ColMajor, float, int64_t>(sycl_device);
  test_sycl_contraction_n<RowMajor, float, int64_t>(sycl_device);
  test_sycl_contraction_k<ColMajor, float, int64_t>(sycl_device);
  test_sycl_contraction_k<RowMajor, float, int64_t>(sycl_device);
  test_sycl_contraction_sizes<ColMajor, float, int64_t>(sycl_device);
  test_sycl_contraction_sizes<RowMajor, float, int64_t>(sycl_device);
  test_TF<RowMajor, float, int64_t>(sycl_device);
  test_TF<ColMajor, float, int64_t>(sycl_device);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

}

EIGEN_DECLARE_TEST(cxx11_tensor_contract_sycl) {
  for (const auto& device :Eigen::get_sycl_supported_devices()) {
    CALL_SUBTEST(tensorContractionPerDevice(device));
  }
}
