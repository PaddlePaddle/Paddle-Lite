// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#define FP32_ABS_DIFF (1e-6)
#define FP32_RELATIVE_DIFF (1e-6)
#define FP16_ABS_DIFF (1e-3)
#define FP16_RELATIVE_DIFF (1e-3)

namespace paddle {
namespace lite {

void softmax_baseline(const float *x_data,
                      float *out_data,
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
          const DDim &x_dim,
          const int axis) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p) << " x_dim=" << x_dim
            << " axis=" << axis;

  auto kernels = KernelRegistry::Global().Create(
      "softmax", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
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
  std::vector<half_t> x_cpu_half(x_dim.production());
  std::vector<float> out_from_cpu(out_dim.production());
  std::vector<float> out_from_gpu(out_dim.production());

  // fill random input
  float *ptr_float = x_cpu.data();
  half_t *ptr_half = x_cpu_half.data();
  fill_data_rand(ptr_float, -1.f, 1.f, x_dim.production());
  if (fp16_flag) {
    for (int i = 0; i < x_dim.production(); i++) {
      ptr_half[i] = Float2Half(ptr_float[i]);
    }
  }

  // x data
  auto *x_data = x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  size_t elemSize = fp16_flag ? sizeof(half_t) : sizeof(float);
  const void *src_ptr = fp16_flag ? reinterpret_cast<const void *>(ptr_half)
                                  : reinterpret_cast<const void *>(ptr_float);
  TargetWrapperCL::MemcpySync(
      x_data, src_ptr, x_dim.production() * elemSize, IoDirection::HtoD);

  // run cpu ref
  softmax_baseline(ptr_float, out_from_cpu.data(), x_dim, axis);

  // run opencl kernel
  kernel->Launch();

#ifdef LITE_WITH_PROFILE
  profile::OpCharacter opchar;
  kernel->SetProfileRuntimeKernelInfo(&opchar);
  double timeInMS = CLRuntime::Global()->GetCommandTime(opchar.cl_event);
  LOG(INFO) << "x_dim=" << x_dim << ", kernel=" << opchar.kernel_func_name
            << ": time cost=" << timeInMS;
#endif

  CLRuntime::Global()->command_queue().finish();

  // output
  auto *out_data = fp16_flag ? out.mutable_data<half_t, cl::Buffer>()
                             : out.mutable_data<float, cl::Buffer>();
  void *out_gpu = out_from_gpu.data();
  TargetWrapperCL::MemcpySync(
      out_gpu, out_data, out_dim.production() * elemSize, IoDirection::DtoH);
  half_t *out_from_gpu_half = reinterpret_cast<half_t *>(out_gpu);

  VLOG(4) << "output_data vs output_ref_data";
  auto relative_diff_thres =
      fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
  auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
  uint32_t diff_cnt = 0;
  for (int i = 0; i < out_dim.production(); i++) {
    float gpu_value =
        fp16_flag ? Half2Float(out_from_gpu_half[i]) : out_from_gpu[i];
    auto relative_diff = COMPUTE_RELATIVE_DIFF(gpu_value, out_from_cpu[i]);
    auto abs_diff = COMPUTE_ABS_DIFF(gpu_value, out_from_cpu[i]);
    EXPECT_FALSE(relative_diff > relative_diff_thres &&
                 abs_diff > abs_diff_thres);
    if (relative_diff > relative_diff_thres && abs_diff > abs_diff_thres) {
      LOG(WARNING) << lite_api::CLPrecisionTypeToStr(p) << "   err idx: " << i
                   << " abs_diff: " << abs_diff
                   << "\t relative_diff: " << relative_diff
                   << "\t out_ins: " << gpu_value
                   << "\t out_ref: " << out_from_cpu[i];
      diff_cnt++;
    }
  }
  if (diff_cnt != 0) {
    LOG(FATAL) << "\n\t[  FAILED  ] "
               << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p)
               << " x_dim=" << x_dim << " axis=" << axis
               << "; diff_cnt= " << diff_cnt << "/" << out_dim.production();
  } else {
    LOG(INFO) << "\n\t[  PASSED  ] "
              << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p)
              << " x_dim=" << x_dim << " axis=" << axis;
  }
}

TEST(softmax, compute_basic) {
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP32,
        lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    for (const auto x_dim : std::vector<std::vector<int64_t>>{
             {31, 11, 53, 831},
             {11, 53, 831},
             {53, 1000},
             {3, 2, 53, 1},
             {2, 53, 1},
             {53, 1},
             {3, 2, 53, 3},
             {2, 53, 3},
             {53, 3},
             {3, 2, 53, 7},
             {2, 53, 7},
             {53, 7},
             {1, 2, 3, 4},
             {2, 3, 4},
             {3, 4},
         }) {
      int ndims = x_dim.size();
      for (int axis = -1; axis < ndims; axis++) {
        test(precision_type, DDim(x_dim), axis);
      }
    }
    // Special case, such as large num
    const auto x_dims = std::vector<int64_t>{64, 1001};
    test(precision_type, DDim(x_dims), 1);
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(softmax, kOpenCL, kFP16, kNCHW, def);
