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

template <typename indtype, typename outdtype>
void argmax_baseline(const indtype* x_data,
                     outdtype* out_data,
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
      std::vector<std::pair<indtype, outdtype>> vec;
      vec.resize(size);
      for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(in_ptr[i * in_stride], i);
      }
      // sort
      std::partial_sort(vec.begin(),
                        vec.begin() + 1,
                        vec.end(),
                        std::greater<std::pair<indtype, outdtype>>());

      // out
      auto* out_ptr = out_data + n * out_channel + k;
      *out_ptr = vec[0].second;
    }
  }
}

void test(const lite_api::CLPrecisionType p,
          const bool keepdims,
          const int axis,
          DDim x_dim) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p) << " axis=" << axis;

  auto kernels = KernelRegistry::Global().Create(
      "arg_max", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());

  lite::Tensor x, out;
  operators::ArgmaxParam param;
  param.X = &x;
  param.Out = &out;
  param.Axis = axis;
  param.keepdims = keepdims;

  kernel->SetParam(param);
  kernel->SetContext(std::move(context));

  std::vector<int64_t> output_shape;
  for (size_t i = 0; i < x_dim.size(); i++) {
    output_shape.push_back(x_dim[i]);
  }
  int axis_new = (axis >= 0) ? axis : axis + x_dim.size();
  output_shape[axis_new] = 1L;
  DDim out_dim(output_shape);

  x.Resize(x_dim);
  out.Resize(out_dim);

  std::vector<float> x_cpu(x_dim.production());
  std::vector<float> out_from_cpu(out_dim.production());
  std::vector<float> out_from_gpu(out_dim.production());
  fill_data_rand(x_cpu.data(), -100.f, 100.f, x_dim.production());

  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim x_image_shape = default_converter->InitImageDimInfoWith(x_dim);
  DDim out_image_shape = default_converter->InitImageDimInfoWith(out_dim);
  VLOG(4) << "x_image_shape = " << x_image_shape[0] << " " << x_image_shape[1];
  VLOG(4) << "out_image_shape = " << out_image_shape[0] << " "
          << out_image_shape[1];

  const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
  std::vector<char> x_image_data(x_image_shape.production() * 4 * dtype_size);
  default_converter->NCHWToImage(x_cpu.data(), x_image_data.data(), x_dim);
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
      out_image_data.data(), out_from_gpu.data(), out_image_shape, out_dim);

  // run cpu ref
  if (fp16_flag) {
    argmax_baseline<float, float>(
        x_cpu.data(), out_from_cpu.data(), x_dim, out_dim, axis_new);
  } else {
    argmax_baseline<float, float>(
        x_cpu.data(), out_from_cpu.data(), x_dim, out_dim, axis_new);
  }

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

void test_argmax_opencl_4d() {
  for (bool keepdims : {true}) {
    for (int axis : {-1, 0, 1, 2, 3}) {
      for (int n : {2, 3}) {
        for (int c : {5, 6}) {
          for (int h : {2, 3, 4, 5, 6}) {
            for (int w : {2, 3, 4, 5, 6}) {
              for (const auto precision_type :
                   {lite_api::CLPrecisionType::CL_PRECISION_FP32}) {
                auto x_dims = DDim(std::vector<int64_t>({n, c, h, w}));
                test(precision_type, keepdims, axis, x_dims);
              }
            }
          }
        }
      }
    }
  }
}

void test_argmax_opencl_3d() {
  for (bool keepdims : {true}) {
    for (int axis : {-1, 0, 1, 2}) {
      for (int c : {4, 4}) {
        for (int h : {2, 10}) {
          for (int w : {2, 17}) {
            for (const auto precision_type :
                 {lite_api::CLPrecisionType::CL_PRECISION_FP32}) {
              auto x_dims = DDim(std::vector<int64_t>({c, h, w}));
              test(precision_type, keepdims, axis, x_dims);
            }
          }
        }
      }
    }
  }
}

void test_argmax_opencl_2d() {
  for (bool keepdims : {true}) {
    for (int axis : {-1, 0, 1}) {
      for (int h : {2, 10}) {
        for (int w : {2, 17}) {
          for (const auto precision_type :
               {lite_api::CLPrecisionType::CL_PRECISION_FP32}) {
            auto x_dims = DDim(std::vector<int64_t>({h, w}));
            test(precision_type, keepdims, axis, x_dims);
          }
        }
      }
    }
  }
}

TEST(argmax, compute_basic) {
  test_argmax_opencl_4d();
  test_argmax_opencl_3d();
  test_argmax_opencl_2d();
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(arg_max, kOpenCL, kFP16, kImageDefault, def);
