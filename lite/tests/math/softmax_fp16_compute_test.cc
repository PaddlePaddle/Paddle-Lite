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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/fill_data.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/tests/utils/print_info.h"
#include "lite/tests/utils/tensor_utils.h"

#ifdef LITE_WITH_ARM
#include "lite/kernels/arm/softmax_compute.h"
#endif  // LITE_WITH_ARM

DEFINE_int32(power_mode,
             3,
             "power mode: "
             "0 for POWER_HIGH;"
             "1 for POWER_LOW;"
             "2 for POWER_FULL;"
             "3 for NO_BIND");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");
DEFINE_bool(basic_test, false, "do all tests");
DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(batch, 1, "batch size");
DEFINE_int32(in_channel, 32, "input channel");
DEFINE_int32(in_height, 112, "input height");
DEFINE_int32(in_width, 112, "input width");

DEFINE_int32(axis, 1, "input width");

typedef paddle::lite::DDim DDim;
typedef paddle::lite::Tensor Tensor;
typedef paddle::lite::operators::SoftmaxParam SoftmaxParam;
using paddle::lite::profile::Timer;

template <typename dtype>
void softmax_compute_ref(const dtype* x_data,
                         dtype* output_data,
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
    dtype max_data = std::numeric_limits<dtype>::lowest();
    for (int j = 0; j < axis_size; j++) {
      max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
      offset += inner_num;
    }

    offset = start;
    dtype sum_data = (dtype)0;
    for (int j = 0; j < axis_size; j++) {
      output_data[offset] = exp(x_data[offset] - max_data);
      sum_data += output_data[offset];
      offset += inner_num;
    }

    offset = start;
    for (int j = 0; j < axis_size; j++) {
      output_data[offset] /= sum_data;
      offset += inner_num;
    }
  }
}

#ifdef LITE_WITH_ARM
void test_softmax_fp16(const DDim in_dim,
                       int axis,
                       const std::vector<int>& thread_num,
                       const std::vector<int>& power_mode) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  SoftmaxParam param;
  param.x = new Tensor;
  param.x->set_precision(PRECISION(kFP16));
  param.axis = axis;

  param.output = new Tensor;
  param.output->set_precision(PRECISION(kFP16));

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::SoftmaxCompute<PRECISION(kFP16),
                                                 PRECISION(kFP16)>
          softmax;
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      /// set param and context
      softmax.SetParam(param);
      softmax.SetContext(std::move(ctx1));
      /// prepare for run
      softmax.PrepareForRun();

      param.x->Resize(in_dim);
      param.output->Resize(in_dim);

      Tensor x_fp32;
      x_fp32.Resize(in_dim);
      x_fp32.set_precision(PRECISION(kFloat));
      auto a_ptr = x_fp32.mutable_data<float>();
      auto b_ptr = param.x->mutable_data<float16_t>();
      // fill_data_rand<float16_t>(b_ptr, -1.f, 1.f, param.x->numel());
      fill_data_const<float16_t>(b_ptr, -1.f, param.x->numel());
      fp16_to_float(param.x->data<float16_t>(), a_ptr, param.x->numel());
      auto din = param.x->data<float16_t>();
      auto din_fp32 = x_fp32.data<float>();

      Tensor tout_basic;
      if (FLAGS_check_result) {
        Tensor tout_basic_fp32;
        tout_basic_fp32.set_precision(PRECISION(kFloat));
        tout_basic.set_precision(PRECISION(kFP16));
        tout_basic_fp32.Resize(in_dim);
        tout_basic.Resize(in_dim);

        auto dout_basic = tout_basic.mutable_data<float16_t>();
        auto dout_basic_fp16 = tout_basic.mutable_data<float16_t>();
        auto dout_basic_fp32 = tout_basic_fp32.mutable_data<float>();
        fill_data_const<float>(dout_basic_fp32, 0.f, tout_basic_fp32.numel());
        softmax_compute_ref<float>(din_fp32, dout_basic_fp32, in_dim, axis);
        // fp32->fp16
        float_to_fp16(dout_basic_fp32, dout_basic, tout_basic.numel());
      }
      /// warm up
      for (int i = 0; i < FLAGS_warmup; ++i) {
        softmax.Launch();
      }
      /// compute
      Timer t0;
      for (int i = 0; i < FLAGS_repeats; ++i) {
        t0.Start();
        softmax.Launch();
        t0.Stop();
      }

      VLOG(4) << "softmax fp16: input shape: " << in_dim << ", axis: " << axis
              << ", running time, avg: " << t0.LapTimes().Avg()
              << ", min time: " << t0.LapTimes().Min();

      if (FLAGS_check_result) {
        double max_ratio = 0;
        double max_diff = 0;
        auto basic_ptr = tout_basic.data<float16_t>();
        auto saber_ptr = param.output->data<float16_t>();
        Tensor tdiff;
        tdiff.Resize(tout_basic.dims());
        tdiff.set_precision(PRECISION(kFP16));
        auto ptr = tdiff.mutable_data<float16_t>();
        data_diff(
            basic_ptr, saber_ptr, ptr, tout_basic.numel(), max_ratio, max_diff);
        print_diff_info(max_diff, max_ratio);
        if (std::abs(max_ratio) > 1e-3f) {
          if (max_diff > 4e-3f) {
            int64_t size = tout_basic.numel();
            int64_t width = in_dim[3];
            print_tensor_info_fp16(basic_ptr, saber_ptr, ptr, size, width);
            LOG(FATAL) << "test fp16 softmax: input: " << in_dim
                       << ", axis: " << axis << ", threads: " << th
                       << ", power_mode: " << cls << " failed!!\n";
          }
        }
      }
      LOG(INFO) << "test fp16 softmax: input: " << in_dim << ", axis: " << axis
                << ", threads: " << th << ", power_mode: " << cls
                << " successed!!\n";
    }
  }

  delete param.x;
  delete param.output;
}

#else
void test_softmax_fp16(const DDim in_dim,
                       int axis,
                       const std::vector<int>& thread_num,
                       const std::vector<int>& power_mode) {}
#endif  // LITE_WITH_ARM

#if 1  /// random param softmax
TEST(TestSoftmaxRand, test_softmax_rand) {
  if (FLAGS_basic_test) {
    for (auto n : {1, 3, 4, 11}) {
      for (auto c : {1, 3, 11, 4}) {
        for (auto h : {3, 1, 11, 4}) {
          for (auto w : {1, 3, 4, 12}) {
            for (auto axis : {-4, -3, -2, -1, 0, 1, 2, 3}) {
              DDim in_dim({n, c, h, w});
              test_softmax_fp16(in_dim, axis, {4}, {FLAGS_power_mode});
            }
          }
        }
      }
    }
  }
}
#endif  /// random param conv

#if 1  /// custom
TEST(TesSoftmaxCustom, test_softmax_fp16_custom_size) {
  test_softmax_fp16(
      DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width}),
      FLAGS_axis,
      {FLAGS_threads},
      {FLAGS_power_mode});
}
#endif  // custom
