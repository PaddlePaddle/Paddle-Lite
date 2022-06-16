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

#define FP16_MAX_DIFF (3e-4)

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

void PrintData(std::string name,
               float* a,
               const int in,
               const int ic,
               const int ih,
               const int iw) {
  std::cout << "==== " << name << " ====" << std::endl;
  for (int n = 0; n < in; ++n) {
    for (int c = 0; c < ic; ++c) {
      for (int h = 0; h < ih; ++h) {
        for (int w = 0; w < iw; ++w) {
          std::cout << " " << a[n * ic * ih * iw + c * ih * iw + h * iw + w];
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

static inline void TestWithKernel(
    const std::unique_ptr<paddle::lite::KernelBase>& kernel) {
  int64_t batch_size = 1;
  int64_t ic = 15;
  int64_t ih = 1;
  int64_t iw = 640;

  int64_t oc = 15;
  int64_t oh = 1;
  int64_t ow = 512;

  lite_api::CLPrecisionType p = lite_api::CLPrecisionType::CL_PRECISION_FP32;
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);

  lite::Tensor input, input_h, output, output_h, cell, hidden, w_0, w_1, b_0,
      b_1, state_1, state_2;
  operators::RnnParam param;
  // opencl param
  if (fp16_flag) {
    param.Input = &input_h;
    param.Out = &output_h;
  } else {
    param.Input = &input;
    param.Out = &output;
  }
  param.WeightList.push_back(&w_0);
  param.WeightList.push_back(&w_1);
  param.WeightList.push_back(&b_0);
  param.WeightList.push_back(&b_1);
  param.PreState.push_back(&cell);
  param.PreState.push_back(&hidden);
  param.State.push_back(&state_1);
  param.State.push_back(&state_2);
  param.hidden_size = 512;
  param.input_size = 640;
  param.is_bidirec = false;
  param.is_test = true;
  param.num_layers = 1;
  param.mode = "LSTM";

  const DDim w0_dim = DDim{std::vector<int64_t>({2048, 640})};
  const DDim w1_dim = DDim{std::vector<int64_t>({2048, 512})};
  const DDim b_dim = DDim{std::vector<int64_t>({2048})};
  const DDim cell_dim = DDim{std::vector<int64_t>({1, 1, 512})};
  const DDim hidden_dim = DDim{std::vector<int64_t>({1, 1, 512})};
  const DDim input_dim = lite::DDim{std::vector<int64_t>({ic, ih, iw})};
  const DDim output_dim = lite::DDim{std::vector<int64_t>({oc, oh, ow})};
  input.Resize(input_dim);
  input_h.Resize(input_dim);
  output.Resize(output_dim);
  output_h.Resize(output_dim);
  state_1.Resize(cell_dim);
  state_2.Resize(cell_dim);

  w_0.Resize(w0_dim);
  w_1.Resize(w1_dim);
  b_0.Resize(b_dim);
  b_1.Resize(b_dim);
  cell.Resize(cell_dim);
  hidden.Resize(hidden_dim);

  LOG(INFO) << "prepare kernel SetParam------";
  kernel->SetParam(param);

  LOG(INFO) << "gen input ...";

  // float* input_v_data = &input_v[0];
  auto index = 0;
  // for (auto& i : input_v) {
  //   i = index++;
  // }
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-5, 5);

  std::vector<float> x_source(input_dim.production());
  std::vector<float> cell_source(cell_dim.production());
  std::vector<float> hidden_source(hidden_dim.production());
  std::vector<half_t> x_source_half(input_dim.production());
  std::vector<float> output_source(output_dim.production());
  std::vector<float> output_half2float(output_dim.production());
  size_t x_size = input_dim.production() * sizeof(float);
  size_t cell_size = cell_dim.production() * sizeof(float);
  for (size_t i = 0; i < input_dim.production(); ++i) {
    x_source[i] = 0.1;
    x_source_half[i] = Float2Half(x_source[i]);
  }
  for (size_t i = 0; i < cell_dim.production(); ++i) {
    cell_source[i] = 0.2;
    hidden_source[i] = 0.3;
  }
  auto* x_data = input.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto* x_data_h = input_h.mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL));
  // auto* cell_data = cell.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  // auto* hidden_data = hidden.mutable_data<float,
  // cl::Buffer>(TARGET(kOpenCL));
  auto* out_data = output.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto* out_data_h = output_h.mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL));
  if (fp16_flag) {
    x_size = input_dim.production() * sizeof(half_t);
    TargetWrapperCL::MemcpySync(
        x_data_h, x_source_half.data(), x_size, IoDirection::HtoD);
  } else {
    TargetWrapperCL::MemcpySync(
        x_data, x_source.data(), x_size, IoDirection::HtoD);
  }
  // TargetWrapperCL::MemcpySync(cell_data, cell_source.data(), cell_size,
  // IoDirection::HtoD);
  // TargetWrapperCL::MemcpySync(hidden_data, hidden_source.data(), cell_size,
  // IoDirection::HtoD);

  std::vector<float> w0_source(w0_dim.production());
  std::vector<float> w1_source(w1_dim.production());
  std::vector<float> b0_source(b_dim.production());
  std::vector<float> b1_source(b_dim.production());
  auto* w0_data = w_0.mutable_data<float>();
  auto* w1_data = w_1.mutable_data<float>();
  auto* b0_data = b_0.mutable_data<float>();
  auto* b1_data = b_1.mutable_data<float>();
  auto* cell_data = cell.mutable_data<float>();
  auto* hidden_data = hidden.mutable_data<float>();
  for (size_t i = 0; i < cell_dim.production(); ++i) {
    cell_data[i] = 0.011;
    hidden_data[i] = 0.022;
  }
  for (size_t i = 0; i < w0_dim.production(); ++i) {
    w0_source[i] = 0.02;
    w0_data[i] = w0_source[i];
  }
  for (size_t i = 0; i < w1_dim.production(); ++i) {
    w1_source[i] = 0.02;
    w1_data[i] = w1_source[i];
  }
  for (size_t i = 0; i < b_dim.production(); ++i) {
    b0_source[i] = 0.03;
    b0_data[i] = b0_source[i];
    b1_source[i] = 0.04;
    b1_data[i] = b1_source[i];
  }

  LOG(INFO) << "kernel context ...";
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();

  std::unique_ptr<KernelContext> transpose_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(transpose_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(transpose_context));

  LOG(INFO) << "kernel launch ...";
  // for (int i = 0; i < 100; i++){
  float time_use = 0;
  struct timeval start;
  struct timeval end;
  gettimeofday(&start, NULL);

  // for (int i = 0; i < 100; i++){
  kernel->Launch();
  CLRuntime::Global()->command_queue().finish();
  // }

  gettimeofday(&end, NULL);
  time_use = (end.tv_sec - start.tv_sec) * 1000000 +
             (end.tv_usec - start.tv_usec);  //微秒
  printf("rnn opencl time_use is %.10f\n", time_use);
  // }

  std::vector<float> out_data_from_gpu(output_dim.production());
  std::vector<half_t> out_data_from_gpu_half(output_dim.production());
  if (fp16_flag) {
    TargetWrapperCL::MemcpySync(out_data_from_gpu_half.data(),
                                out_data_h,
                                out_data_from_gpu_half.size() * sizeof(half_t),
                                IoDirection::DtoH);
  } else {
    TargetWrapperCL::MemcpySync(out_data_from_gpu.data(),
                                out_data,
                                out_data_from_gpu.size() * sizeof(float),
                                IoDirection::DtoH);
  }

  for (int eidx = 0; eidx < output_dim.production(); ++eidx) {
    output_half2float[eidx] = Half2Float(out_data_from_gpu_half.data()[eidx]);
  }

  // PrintData("input", static_cast<float*>(x_source.data()), batch_size, ic,
  // ih, iw);
  // PrintData("output", static_cast<float*>(output_source.data()), batch_size,
  // oc, oh, ow);
  PrintData("gpu",
            static_cast<float*>(out_data_from_gpu.data()),
            batch_size,
            oc,
            oh,
            ow);
  // PrintData("gpu_half", static_cast<float*>(output_half2float.data()),
  // batch_size, oc, oh, ow);
  // check output data
  index = 0;
  // for (auto n = 0; n < batch_size; n++) {
  //   for (auto h = 0; h < ih; h++) {
  //     for (auto c = 0; c < ic; c++) {
  //       for (auto w = 0; w < iw; w++) {
  //         auto input_index = n * ic * ih * iw + c * ih * iw + h * iw + w;
  //         auto input_value = x_source[input_index];
  //         float output_value = 0.f;
  //         if (fp16_flag){
  //           output_value = Half2Float(out_data_from_gpu_half.data()[index]);
  //         } else {
  //           output_value = out_data_from_gpu[index];
  //         }
  //         auto abs_diff = abs(input_value - output_value);
  //         auto relative_diff = COMPUTE_RELATIVE_DIFF(input_value,
  //         output_value);
  //         EXPECT_EQ(
  //             (relative_diff <= FP16_MAX_DIFF) || (abs_diff <=
  //             FP16_MAX_DIFF),
  //             true);
  //         // if (relative_diff > FP16_MAX_DIFF){
  //         //   std::cout << "output_value: " << output_value << ";
  //         input_value: " << input_value << std::endl;
  //         // }
  //         index++;
  //       }
  //     }
  //   }
  // }
}

TEST(rnn_opencl, compute) {
  auto kernels = KernelRegistry::Global().Create(
      "rnn", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  TestWithKernel(kernel);
}

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(rnn, kOpenCL, kFP16, kNCHW, def);
