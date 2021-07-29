// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <iostream>
#include <algorithm>
#include <random>
#include <ctime>
#include <cmath>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

namespace paddle {
namespace lite {

template <typename Dtype>
void fill_data(Dtype *x, const int n, const int c, const int h, const int w, int set_value = -1) {
  if (set_value == -1) {
    int length = n * c * h * w;
    std::default_random_engine engine(time(0));
    std::uniform_real_distribution<float> gen(-2, 2);
    for (size_t idx = 0; idx < length; ++idx) {
      x[idx] = gen(engine);
    }
  } else if (set_value == 1) {
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < c; ++j) {
        for (size_t k = 0; k < h; ++k) {
          for (size_t l = 0; l < w; ++l) {
            x[i * (c * h * w) + j * (h * w) + k * w + l] = (k == l) ? 2 : 0;
          }
        }
      }
    }
  } else {
    int length = n * c * h * w;
    for (size_t idx = 0; idx < length; ++idx) {
      x[idx] = set_value;
    }
  }
}

template <typename Dtype>
void MulAdd(Dtype* x, int i, int j, Dtype a, int cols) {
  for (int k = 0; k < cols; k++) {
    x[j * cols + k] += (a * x[i * cols + k]);
  }
}

template <typename Dtype>
void swap2row(Dtype* x, int cols, int i, int j) {
  for (int k = 0; k < cols; k++) {
    Dtype temp = x[i * cols + k];
    x[i * cols + k] = x[j * cols + k];
    x[j * cols + k] = temp;
  }
}

template <typename Dtype>
void inverse_compute_ref(const Dtype* x, Dtype* out, int rows, int cols) { // Gauss elimination
  Dtype* temp = new Dtype[2 * rows * cols];
  for (int i = 0; i < 2 * rows * cols; i++) temp[i] = 0;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      temp[i * 2 * cols + j] = x[i * cols + j];         
    }
    temp[i * 2 * cols + cols + i] = 1;
  }
  for (int i = 0; i < rows; i++) {
    Dtype a1 = temp[i * 2 * cols + i];
    int swap_i = i;
    for (int k = i + 1; k < rows; k++) {
      if (std::abs(temp[k * 2 * cols + i] > std::abs(a1))) {
        swap_i = k;
        a1 = temp[k * 2 * cols + i];
      }
    }
    if (swap_i != i) {
      for (int k = 0; k < 2 * cols; k++) {
        Dtype tmp = temp[swap_i * 2 * cols + k];
        temp[swap_i * 2 * cols + k] = temp[i * 2 * cols + k];
        temp[i * 2 * cols + k] = tmp;
      }
    }
    for (int k = i; k < 2 * cols; k++) {
      temp[i * 2 * cols + k] /= a1;
    }

    for (int j = 0; j < rows; j++) {
      if (j != i) {
        Dtype a2 = temp[j * 2 * cols + i];
        for (int k = 0; k < 2 * cols; k++) {
          temp[j * 2 * cols + k] += ((-a2) * temp[i * 2 * cols + k]);
        }
      }
    }
  }

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      out[i * cols + j] = temp[i * 2 * cols + cols + j];
    }
  }
  delete[] temp;
  temp = nullptr;
}

TEST(inverse_buffer, compute) {
  const int n = 1;
  const int c = 4;//8;//
  const int h = 4;//8;//
  const int w = 4;//8;//

  const DDim x_dim = DDim(std::vector<DDim::value_type>{n, c, h, w});
  auto out_dim = x_dim;

  LOG(INFO) << "================== inverse ===================";
  lite::Tensor inverse_x, inverse_out;
  inverse_x.Resize(x_dim);
  inverse_out.Resize(out_dim);
  VLOG(4) << "initialize tensors";

  auto *x_data = inverse_x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto *mapped_x = static_cast<float *>(
      TargetWrapperCL::Map(x_data, 0, sizeof(float) * x_dim.production()));
  fill_data<float>(mapped_x, n, c, h, w, -1);
  std::vector<float> x_v(x_dim.production());
  std::cout << "initialize input data:" << std::endl;
  for (int i = 0; i < x_dim.production(); ++i) {
    x_v.data()[i] = mapped_x[i];
  }

  auto *out_data = inverse_out.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto *mapped_out = static_cast<float *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * out_dim.production()));

  // set kernel
  auto inverse_buf_kernels =
      KernelRegistry::Global().Create("inverse",
                                      TARGET(kOpenCL),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW));
  ASSERT_FALSE(inverse_buf_kernels.empty());
  auto kernel = std::move(inverse_buf_kernels.front());
  VLOG(4) << "get inverse kernel: " << kernel->doc();

  // set context and kernel args
  VLOG(4) << "set context and kernel args";
  operators::InverseParam inverseParam;
  inverseParam.Input = &inverse_x;
  inverseParam.Output = &inverse_out;
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  kernel->SetParam(inverseParam);
  std::unique_ptr<KernelContext> inverse_buf_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(&(inverse_buf_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(inverse_buf_context));

  // run kernel
  VLOG(4) << "run kernel";
  kernel->Launch();

  CLRuntime::Global()->command_queue().finish();

  // compute cpu reference
  std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
  for (int i = 0; i < n * c; i++) {
    inverse_compute_ref<float>(x_v.data() + i * h * w, out_ref.get() + i * h * w, h, w);
  }

  // compare cpu gpu results
#if 0
  std::cout << "gpu out results: " << std::endl;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < c; ++j) {
      for (size_t k = 0; k < h; ++k) {
        for (size_t l = 0; l < w; ++l) {
          std::cout << mapped_out[i * (c * h * w) + j * (h * w) + k * w + l] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "cpu in results: " << std::endl;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < c; ++j) {
      for (size_t k = 0; k < h; ++k) {
        for (size_t l = 0; l < w; ++l) {
          std::cout << x_v.data()[i * (c * h * w) + j * (h * w) + k * w + l] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << "cpu out results: " << std::endl;
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < c; ++j) {
      for (size_t k = 0; k < h; ++k) {
        for (size_t l = 0; l < w; ++l) {
          std::cout << out_ref.get()[i * (c * h * w) + j * (h * w) + k * w + l] << " ";
          // std::cout << mapped_x[i * (c * h * w) + j * (h * w) + k * w + l] << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
#endif
  for (int eidx = 0; eidx < out_dim.production(); eidx++) {
    auto value = mapped_out[eidx];
    auto ref_value = out_ref.get()[eidx];
    auto diff = abs(value - ref_value);
    if (ref_value == 0.f) {
      if (diff > 1e-3) {
        std::cout << "diff in this case at eidx[from 0]:" << eidx << " / "
                << out_dim.production() << ", value[" << eidx << "]:" << value
                << ", ref_value[" << eidx << "]:" << ref_value << std::endl;
      }
    } else {
        if (diff / ref_value > 1e-2) {
          std::cout << "diff in this case at eidx[from 0]:" << eidx << " / "
                  << out_dim.production() << ", value[" << eidx << "]:" << value
                  << ", ref_value[" << eidx << "]:" << ref_value << std::endl;
        }
    }
  }
  TargetWrapperCL::Unmap(x_data, mapped_x);
  TargetWrapperCL::Unmap(out_data, mapped_out);
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(inverse, kOpenCL, kFloat, kNCHW, def);
