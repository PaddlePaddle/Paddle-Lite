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
#include <memory>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"
#include "lite/tests/utils/fill_data.h"

#define FP16_MAX_DIFF (5e-1)
#define FP32_ABS_DIFF (1e-7)
#define FP32_RELATIVE_DIFF (1e-6)
#define FP16_ABS_DIFF (1e-3)
#define FP16_RELATIVE_DIFF (1e-3)
namespace paddle {
namespace lite {

template <typename indtype>
void max_baseline_dim_single(const float* x_data,
                             float* out_data,
                             const DDim input_dims,
                             const DDim output_dims,
                             int axis) {
  const int size = input_dims[axis];
  const int in_channel = input_dims.count(axis, input_dims.size());
  const int out_channel = output_dims.count(axis, output_dims.size());
  const int in_stride = input_dims.count(axis + 1, input_dims.size());
  const int out_stride = input_dims.count(0, axis);

  for (int n = 0; n < out_stride; n++) {
    for (int k = 0; k < in_stride; k++) {
      const indtype* in_ptr = x_data + n * in_channel + k;
      std::vector<indtype> vec;
      vec.resize(size);
      for (int i = 0; i < size; i++) {
        vec[i] = in_ptr[i * in_stride];
      }
      // sort
      std::partial_sort(
          vec.begin(), vec.begin() + 1, vec.end(), std::greater<indtype>());

      // out
      auto* out_ptr = out_data + n * out_channel + k;
      *out_ptr = vec[0];
    }
  }
}

template <typename indtype>
void max_baseline(const float* x_data,
                  float* out_data,
                  const DDim input_dims,
                  const DDim output_dims,
                  std::vector<int> dim) {
  lite::Tensor tin_tmp;
  lite::Tensor tout_tmp;
  tin_tmp.Resize(input_dims);
  tout_tmp.Resize(input_dims);
  float* tmp_in = tin_tmp.mutable_data<float>();
  float* tmp_out = tout_tmp.mutable_data<float>();
  DDim in_dim = input_dims;
  DDim out_dim = input_dims;
  std::vector<int> real_dim = dim;
  if (dim.size() == 0) {
    real_dim.resize(input_dims.size());
    for (int i = 0; i < real_dim.size(); ++i) {
      real_dim[i] = i;
    }
  }
  for (size_t i = 0; i < real_dim.size(); i++) {
    const float* input_data = (i == 0) ? x_data : tmp_in;
    float* output_data = (i == real_dim.size() - 1) ? out_data : tmp_out;
    out_dim[real_dim[i]] = 1;
    max_baseline_dim_single<float>(
        input_data, output_data, in_dim, out_dim, real_dim[i]);
    std::swap(tmp_in, tmp_out);
    in_dim = out_dim;
  }
}

void max_test(const lite_api::CLPrecisionType p,
              std::vector<int> dim,
              bool keepdims,
              DDim x_dims) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p);

  auto kernels = KernelRegistry::Global().Create(
      "max", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());

  bool reduce_all = false;
  auto x_rank = x_dims.size();
  lite::Tensor x, out;
  x.Resize(x_dims);
  if (!dim.empty()) {
    for (size_t i = 0; i < dim.size(); i++) {
      if (dim[i] < 0) {
        dim[i] += x_rank;
      }
    }
  }

  std::stable_sort(dim.begin(), dim.end());
  if (dim.size() == 0) {
    reduce_all = true;
  }

  std::vector<int64_t> out_dims_shape;
  // DDim out_dims;
  if (reduce_all) {
    if (keepdims) {
      for (size_t i = 0; i < x_dims.size(); i++) {
        out_dims_shape.push_back(1);
      }
    } else {
      out_dims_shape.push_back(1);
    }
  } else {
    for (size_t i = 0; i < x_dims.size(); i++) {
      out_dims_shape.push_back(x_dims[i]);
    }
    if (keepdims) {
      for (size_t i = 0; i < dim.size(); ++i) {
        out_dims_shape[dim[i]] = 1L;
      }
    } else {
      int64_t kDelFlag = -2;
      for (size_t i = 0; i < dim.size(); ++i) {
        out_dims_shape[dim[i]] = kDelFlag;
      }
      out_dims_shape.erase(
          remove(out_dims_shape.begin(), out_dims_shape.end(), kDelFlag),
          out_dims_shape.end());
    }
    if (!keepdims && out_dims_shape.empty()) {
      out_dims_shape.push_back(1);
    }
    out.Resize(DDim(out_dims_shape));
  }

  DDim out_dims = DDim(out_dims_shape);
  operators::ReduceParam param;
  param.X = &x;
  param.Out = &out;
  param.dim = dim;
  param.keep_dim = keepdims;
  param.reduce_all = reduce_all;

  kernel->SetParam(param);
  kernel->SetContext(std::move(context));

  std::vector<float> x_cpu(x_dims.production());
  std::vector<float> out_from_cpu(out_dims.production());
  std::vector<float> out_from_gpu(out_dims.production());
  fill_data_rand(x_cpu.data(), -1.f, 1.f, x_dims.production());

  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim x_image_shape = default_converter->InitImageDimInfoWith(x_dims);
  DDim out_image_shape = default_converter->InitImageDimInfoWith(out_dims);
  VLOG(4) << "x_image_shape = " << x_image_shape[0] << " " << x_image_shape[1];
  VLOG(4) << "out_image_shape = " << out_image_shape[0] << " "
          << out_image_shape[1];

  const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
  std::vector<char> x_image_data(x_image_shape.production() * 4 * dtype_size);
  default_converter->NCHWToImage(x_cpu.data(), x_image_data.data(), x_dims);
  MUTABLE_DATA_GPU(&x, x_image_shape[0], x_image_shape[1], x_image_data.data());
  auto* out_image =
      MUTABLE_DATA_GPU(&out, out_image_shape[0], out_image_shape[1], nullptr);

  // run opencl kernel
  kernel->Launch();
  CLRuntime::Global()->command_queue().finish();

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  std::vector<char> out_image_data(out_image_shape.production() * 4 *
                                   dtype_size);  // 4 : RGBA
  TargetWrapperCL::ImgcpySync(out_image_data.data(),
                              out_image,
                              out_image_shape[0],
                              out_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  default_converter->ImageToNCHW(
      out_image_data.data(), out_from_gpu.data(), out_image_shape, out_dims);

  // run cpu ref
  max_baseline<float>(x_cpu.data(), out_from_cpu.data(), x_dims, out_dims, dim);

  VLOG(4) << "output_data vs output_ref_data";
  auto relative_diff_thres =
      fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
  auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
  uint32_t diff_cnt = 0;
  for (int i = 0; i < out_dims.production(); i++) {
    auto relative_diff =
        COMPUTE_RELATIVE_DIFF(out_from_gpu[i], out_from_cpu[i]);
    auto abs_diff = COMPUTE_ABS_DIFF(out_from_gpu[i], out_from_cpu[i]);
    EXPECT_FALSE(relative_diff > relative_diff_thres &&
                 abs_diff > abs_diff_thres);
    if (relative_diff > relative_diff_thres && abs_diff > abs_diff_thres) {
      LOG(WARNING) << lite_api::CLPrecisionTypeToStr(p) << "   err idx: " << i
                   << " abs_diff: " << abs_diff
                   << "\t relative_diff: " << relative_diff
                   << "\t out_ins: " << out_from_gpu[i]
                   << "\t out_ref: " << out_from_cpu[i];
      diff_cnt++;
    }
  }
  if (diff_cnt != 0) {
    LOG(FATAL) << "Err num " << diff_cnt << "/" << out_dims.production();
  }

  LOG(INFO) << "\n\t[  PASSED  ] "
            << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p)
            << " x_dim=" << x_dims;
}

void test_max_opencl_4d() {
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP32,
        lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    std::vector<std::vector<int>> reduce_dim{{},
                                             {0},
                                             {1},
                                             {2},
                                             {3},
                                             {0, 1},
                                             {1, 2},
                                             {2, 3},
                                             {-2, -1},
                                             {0, 1, 2},
                                             {1, 2, 3}};
    for (int n : {1, 3}) {
      for (int c : {1, 3}) {
        for (int h : {1, 3}) {
          for (int w : {1, 3}) {
            for (auto dim : reduce_dim) {
              auto x_dims = DDim(std::vector<int64_t>({n, c, h, w}));
              max_test(precision_type, dim, true, x_dims);
            }
          }
        }
      }
    }
  }
}

void test_max_opencl_3d() {
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP32,
        lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    std::vector<std::vector<int>> reduce_dim{{}, {0}, {1}, {2}, {0, 1}, {1, 2}};
    for (int c : {1, 3}) {
      for (int h : {1, 3}) {
        for (int w : {1, 3}) {
          for (auto dim : reduce_dim) {
            auto x_dims = DDim(std::vector<int64_t>({c, h, w}));
            max_test(precision_type, dim, true, x_dims);
          }
        }
      }
    }
  }
}

void test_max_opencl_2d() {
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP32,
        lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    std::vector<std::vector<int>> reduce_dim{{}, {0}, {1}};
    for (int h : {2, 3}) {
      for (int w : {2, 3}) {
        for (auto dim : reduce_dim) {
          auto x_dims = DDim(std::vector<int64_t>({h, w}));
          max_test(precision_type, dim, true, x_dims);
        }
      }
    }
  }
}

TEST(max, compute_basic) {
  test_max_opencl_4d();
  test_max_opencl_3d();
  test_max_opencl_2d();
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(reduce_max, kOpenCL, kFP16, kImageDefault, image2d);
