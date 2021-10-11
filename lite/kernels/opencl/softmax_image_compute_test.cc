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
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"
#include "lite/tests/utils/fill_data.h"

#define FP32_ABS_DIFF (1e-7)
#define FP32_RELATIVE_DIFF (1e-6)
#define FP16_ABS_DIFF (1e-3)
#define FP16_RELATIVE_DIFF (1e-3)

namespace paddle {
namespace lite {

void softmax_baseline(const float* x_data,
                      float* out_data,
                      const DDim x_dims,
                      int axis) {
  auto x_rank = x_dims.size();
  if (axis < 0) {
    axis += x_rank;
  }
  int axis_size = x_dims[axis];
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int compute_size = outer_num * inner_num;
  for (int i = 0; i < compute_size; i++) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int start = idx_outer * inner_num + idx_inner;
    int offset;

    offset = start;
    float max_data = std::numeric_limits<float>::lowest();
    for (int j = 0; j < axis_size; j++) {
      max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
      offset += inner_num;
    }

    offset = start;
    float sum_data = 0.f;
    for (int j = 0; j < axis_size; j++) {
      out_data[offset] = exp(x_data[offset] - max_data);
      sum_data += out_data[offset];
      offset += inner_num;
    }

    offset = start;
    for (int j = 0; j < axis_size; j++) {
      out_data[offset] /= sum_data;
      offset += inner_num;
    }
  }
}

void test(const lite_api::CLPrecisionType p,
          const DDim& x_dim,
          const int axis) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p) << " x_dim=" << x_dim
            << " axis=" << axis;

  auto kernels = KernelRegistry::Global().Create(
      "softmax", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageFolder));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());

  lite::Tensor x, out;
  operators::SoftmaxParam param;
  param.x = &x;
  param.output = &out;
  param.axis = axis;

  kernel->SetParam(param);
  kernel->SetContext(std::move(context));

  DDim out_dim = x_dim;
  x.Resize(x_dim);
  out.Resize(out_dim);

  std::vector<float> x_cpu(x_dim.production());
  std::vector<float> out_from_cpu(out_dim.production());
  std::vector<float> out_from_gpu(out_dim.production());
  fill_data_rand(x_cpu.data(), -1.f, 1.f, x_dim.production());

  CLImageConverterFolder* folder_converter = new CLImageConverterFolder();
  DDim x_image_shape = folder_converter->InitImageDimInfoWith(x_dim);
  DDim out_image_shape = folder_converter->InitImageDimInfoWith(out_dim);
  VLOG(4) << "x_image_shape = " << x_image_shape[0] << " " << x_image_shape[1];
  VLOG(4) << "out_image_shape = " << out_image_shape[0] << " "
          << out_image_shape[1];

  const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
  std::vector<char> x_image_data(x_image_shape.production() * 4 * dtype_size);
  folder_converter->NCHWToImage(x_cpu.data(), x_image_data.data(), x_dim);
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
  folder_converter->ImageToNCHW(
      out_image_data.data(), out_from_gpu.data(), out_image_shape, out_dim);

  // run cpu ref
  softmax_baseline(x_cpu.data(), out_from_cpu.data(), x_dim, axis);

  VLOG(4) << "output_data vs output_ref_data";
  auto relative_diff_thres =
      fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
  auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
  uint32_t diff_cnt = 0;
  for (int i = 0; i < out_dim.production(); i++) {
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
    LOG(FATAL) << "Err num " << diff_cnt << "/" << out_dim.production();
  }

  LOG(INFO) << "\n\t[  PASSED  ] "
            << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p)
            << " x_dim=" << x_dim << " axis=" << axis;
}

TEST(softmax, compute_basic) {
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP32,
        lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    for (const auto x_dim :
         std::vector<std::vector<int64_t>>{{1, 2, 3, 4}, {2, 3, 4}, {3, 4}}) {
      int ndims = x_dim.size();
      for (int axis = -1; axis < ndims; axis++) {
        if (axis == 0) continue;
        test(precision_type, DDim(x_dim), axis);
      }
    }

    // Special case, such as large num
    const auto x_dims = std::vector<int64_t>{1, 1000};
    test(precision_type, DDim(x_dims), 1);
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(softmax, kOpenCL, kFP16, kImageFolder, def);
