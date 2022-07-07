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
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"
#include "lite/operators/reshape_op.h"
#include "lite/utils/log/logging.h"

#define FP16_MAX_DIFF (5e-1)

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {
static DDim ValidateShape(const std::vector<int>& shape,
                          const DDim& input_dims) {
  const lite::DDim::value_type input_size = input_dims.production();
  auto input_shape = input_dims.Vectorize();
  bool all_positive = std::all_of(
      input_shape.cbegin(), input_shape.cend(), [](lite::DDim::value_type i) {
        return i > 0;
      });
  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int unk_dim_val = -1;
  const int copy_dim_val = 0;

  std::vector<lite::DDim::value_type> output_shape(shape.size(), 0);
  lite::DDim::value_type capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      CHECK_EQ(unk_dim_idx, -1)
          << "Only one input dimension of Attr(shape) can be unknown.";
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      CHECK_LT(static_cast<int>(i), input_shape.size())
          << "The index of dimension to copy from input shape must be less "
             "than the size of input shape.";
    } else {
      CHECK_GT(shape[i], 0) << "Each input dimension of Attr(shape) must not "
                               "be negtive except one unknown dimension.";
    }

    capacity *= (shape[i] ? static_cast<lite::DDim::value_type>(shape[i])
                          : input_shape[i]);
    output_shape[i] = (shape[i] ? static_cast<lite::DDim::value_type>(shape[i])
                                : input_shape[i]);
  }

  if (unk_dim_idx != -1) {
    if (all_positive) {
      // input_size < 0 and is un-determinate in compile time, skip the check,
      // for example, input_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, input_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_shape[unk_dim_idx] = -input_size / capacity;
      CHECK_EQ(output_shape[unk_dim_idx] * capacity, -input_size)
          << "Invalid shape is given.";
    } else {
      output_shape[unk_dim_idx] = -1;
    }
  } else {
    CHECK_EQ(capacity, input_size) << "Invalid shape is given.";
  }
  return lite::DDim(output_shape);
}

TEST(reshape_opencl, compute) {
  LOG(INFO) << "to get kernel ...";
  auto kernels = KernelRegistry::Global().Create(
      "reshape", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  lite_api::CLPrecisionType p = lite_api::CLPrecisionType::CL_PRECISION_FP16;
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  LOG(INFO) << "created reshape kernel";

  LOG(INFO) << "prepare kernel ------";

  int64_t batch_size = 15;
  int64_t ic = 1;
  int64_t ih = 2;
  int64_t iw = 3;

  lite::Tensor input, output, input_h;

  operators::ReshapeParam param;

  Tensor shape_tensor;
  shape_tensor.Resize({3});
  auto* shape_tensor_data = shape_tensor.mutable_data<int>();
  shape_tensor_data[0] = 1;
  shape_tensor_data[1] = 15;
  shape_tensor_data[2] = 6;

  if (fp16_flag) {
    param.x = &input_h;
    param.shape_tensor = &shape_tensor;  // use shape_tensor
    param.inplace = true;
    param.output = &output;
  } else {
    param.x = &input;
    param.shape_tensor = &shape_tensor;  // use shape_tensor
    param.inplace = true;
    param.output = &output;
  }

  const DDim input_dim =
      lite::DDim{std::vector<int64_t>({batch_size, ic, ih, iw})};
  input.Resize(input_dim);
  input_h.Resize(input_dim);

  std::vector<int> final_shape = std::vector<int>(
      shape_tensor_data, shape_tensor_data + shape_tensor.numel());
  LOG(INFO) << "shape_tensor.numel() " << shape_tensor.numel();
  auto out_dim = ValidateShape(final_shape, input_dim);
  param.output->Resize(out_dim);
  LOG(INFO) << " out_dim------" << out_dim;

  LOG(INFO) << "prepare kernel SetParam------";
  kernel->SetParam(param);
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  kernel->SetContext(std::move(context));

  auto* input_data_h =
      input_h.mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL));
  auto* input_data = input.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);
  LOG(INFO) << "gen input ...";
  std::vector<float> x_source(input_dim.production());
  std::vector<half_t> x_source_half(input_dim.production());
  for (size_t i = 0; i < input_dim.production(); ++i) {
    x_source[i] = static_cast<int>(dist(engine));
    x_source_half[i] = Float2Half(x_source[i]);
  }

  size_t x_size = input_dim.production() * sizeof(float);
  if (fp16_flag) {
    x_size = input_dim.production() * sizeof(half_t);
    TargetWrapperCL::MemcpySync(
        input_data_h, x_source_half.data(), x_size, IoDirection::HtoD);
  } else {
    TargetWrapperCL::MemcpySync(
        input_data, x_source.data(), x_size, IoDirection::HtoD);
  }

  kernel->Launch();
  CLRuntime::Global()->command_queue().finish();
  auto* y_buffer = fp16_flag ? output.data<half_t, cl::Buffer>()
                             : output.data<float, cl::Buffer>();
  std::vector<float> out_data_from_gpu(out_dim.production());
  std::vector<float> output_half2float(out_dim.production());
  std::vector<half_t> out_data_from_gpu_half(out_dim.production());
  if (fp16_flag) {
    TargetWrapperCL::MemcpySync(out_data_from_gpu_half.data(),
                                y_buffer,
                                out_data_from_gpu_half.size() * sizeof(half_t),
                                IoDirection::DtoH);
  } else {
    TargetWrapperCL::MemcpySync(out_data_from_gpu.data(),
                                y_buffer,
                                out_data_from_gpu.size() * sizeof(float),
                                IoDirection::DtoH);
  }
  for (int eidx = 0; eidx < out_dim.production(); ++eidx) {
    output_half2float[eidx] = Half2Float(out_data_from_gpu_half.data()[eidx]);
  }

  // check output dims
  for (int i = 0; i < output.dims().size(); i++) {
    CHECK_EQ(output.dims()[i], shape_tensor_data[i]);
  }

  // check output data
  for (int i = 0; i < output.numel(); i++) {
    auto out_gpu_data = out_data_from_gpu[i];
    if (fp16_flag) {
      out_gpu_data = output_half2float[i];
    }
    auto abs_diff = abs(out_gpu_data - x_source[i]);
    auto relative_diff = COMPUTE_RELATIVE_DIFF(out_gpu_data, x_source[i]);
    EXPECT_EQ((relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
              true);
    if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
      LOG(ERROR) << "error idx:" << i << " out_gpu_data[" << i
                 << "]:" << out_gpu_data << " "
                                            "input_data["
                 << i << "]:" << x_source[i] << " abs_diff:" << abs_diff
                 << " relative_diff:" << relative_diff
                 << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
    }
  }
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(reshape, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(reshape2, kOpenCL, kFloat, kNCHW, def);
