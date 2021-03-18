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
typedef paddle::lite::operators::ElementwiseParam ElementwiseParam;
using paddle::lite::profile::Timer;

#define ELT(MATHOP)                                                          \
  for (int n = 0; n < xn; n++) {                                             \
    for (int c = 0; c < xc; c++) {                                           \
      for (int h = 0; h < xh; h++) {                                         \
        for (int w = 0; w < xw; w++) {                                       \
          int x_offset = n * xc * xh * xw + c * xh * xw + h * xw + w;        \
          int y_offset = 0;                                                  \
          if (yn != 1) y_offset += n * yc * yh * yw;                         \
          if (yc != 1) y_offset += c * yh * yw;                              \
          if (yh != 1) y_offset += h * yw;                                   \
          if (yw != 1) y_offset += w;                                        \
          out_data[x_offset] = MATHOP(out_data[x_offset], y_data[y_offset]); \
        }                                                                    \
      }                                                                      \
    }                                                                        \
  }

template <class T>
T add(T a, T b) {
  return a + b;
}

template <class T>
T sub(T a, T b) {
  return a - b;
}

template <class T>
T mul(T a, T b) {
  return a * b;
}

template <class T>
T div(T a, T b) {
  return a / b;
}

template <class T>
T floordiv(T a, T b) {
  return static_cast<T>(std::trunc(a / b));
}

template <class T>
T max(T a, T b) {
  return std::max(a, b);
}

template <class T>
T min(T a, T b) {
  return std::min(a, b);
}

template <class T>
T pow(T a, T b) {
  return std::pow(a, b);
}

template <class T>
T mod(T a, T b) {
  T res = a % b;
  if ((res != 0) && ((res < 0) != (b < 0))) res += b;
  return res;
}

template <>
float mod<float>(float a, float b) {
  float res = fmod(a, b);
  if ((res != 0) && ((b < 0) != (res < 0))) res += b;
  return res;
}

template <typename dtype>
void ele_add_compute_ref(const dtype* x_data,
                         const dtype* y_data,
                         dtype* out_data,
                         DDim x_dims,
                         DDim y_dims,
                         int axis,
                         std::string elt_type = "add") {
  DDim y_dims = x_dims;
  if (axis < 0) {
    axis = x_dims.size() - y_dims.size();
  }
  auto x_shape = x_dims.Vectorize();
  while (x_shape.size() < 4) {
    x_shape.push_back(1);
  }
  auto y_shape = y_dims.Vectorize();
  y_shape.insert(y_shape.begin(), axis, 1);
  while (y_shape.size() < 4) {
    y_shape.push_back(1);
  }
  CHECK_EQ(x_shape.size(), 4);
  CHECK_EQ(y_shape.size(), 4);

  int xn = x_shape[0];
  int xc = x_shape[1];
  int xh = x_shape[2];
  int xw = x_shape[3];

  int yn = y_shape[0];
  int yc = y_shape[1];
  int yh = y_shape[2];
  int yw = y_shape[3];

  if (elt_type == "add") {
    ELT(add);
  } else if (elt_type == "sub") {
    ELT(sub);
  } else if (elt_type == "mul") {
    ELT(mul);
  } else if (elt_type == "div") {
    ELT(div);
  } else if (elt_type == "floordiv") {
    ELT(floordiv);
  } else if (elt_type == "max") {
    ELT(max);
  } else if (elt_type == "min") {
    ELT(min);
  } else if (elt_type == "pow") {
    ELT(pow);
  } else if (elt_type == "mod") {
    ELT(mod);
  } else {
    LOG(FATAL) << "unsupported: " << elt_type;
  }
}
#ifdef LITE_WITH_ARM
void test_elementwise_fp16(const DDim x_dim,
                           const DDim y_dim,
                           const int axis,
                           const std::vector<int>& thread_num,
                           const std::vector<int>& power_mode) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  SoftmaxParam param;
  param.X = new Tensor;
  param.X->set_precision(PRECISION(kFP16));
  param.Y = new Tensor;
  param.Y->set_precision(PRECISION(kFP16));
  param.axis = axis;

  param.Out = new Tensor;
  param.Out->set_precision(PRECISION(kFP16));

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::ElementwiseAddCompute<PRECISION(kFP16),
                                                        PRECISION(kFP16)>
          ele_add;
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      /// set param and context
      ele_add.SetParam(param);
      ele_add.SetContext(std::move(ctx1));
      /// prepare for run
      ele_add.PrepareForRun();

      param.X->Resize(x_dim);
      param.Y->Resize(y_dim);
      param.Out->Resize(x_dim);

      Tensor x_fp32;
      Tensor y_fp32;
      x_fp32.Resize(x_dim);
      x_fp32.set_precision(PRECISION(kFloat));
      y_fp32.Resize(y_dim);
      y_fp32.set_precision(PRECISION(kFloat));
      auto a_ptr = x_fp32.mutable_data<float>();
      auto b_ptr = param.X->mutable_data<float16_t>();
      fill_data_rand<float16_t>(b_ptr, -1.f, 1.f, param.X->numel());
      // fill_data_const<float16_t>(b_ptr, -1.f, param.X->numel());
      fp16_to_float(param.X->data<float16_t>(), a_ptr, param.X->numel());
      a_ptr = y_fp32.mutable_data<float>();
      b_ptr = param.Y->mutable_data<float16_t>();
      fill_data_rand<float16_t>(b_ptr, -1.f, 1.f, param.Y->numel());
      fp16_to_float(param.Y->data<float16_t>(), a_ptr, param.Y->numel());
      auto dinx_fp32 = x_fp32.data<float>();
      auto diny_fp32 = y_fp32.data<float>();

      Tensor tout_basic;
      if (FLAGS_check_result) {
        Tensor tout_basic_fp32;
        tout_basic_fp32.set_precision(PRECISION(kFloat));
        tout_basic.set_precision(PRECISION(kFP16));
        tout_basic_fp32.Resize(x_dim);
        tout_basic.Resize(x_dim);

        auto dout_basic = tout_basic.mutable_data<float16_t>();
        auto dout_basic_fp32 = tout_basic_fp32.mutable_data<float>();
        fill_data_const<float>(dout_basic_fp32, 0.f, tout_basic_fp32.numel());
        ele_add_compute_ref<float>(
            dinx_fp32, diny_fp32, dout_basic_fp32, x_dim, y_dim, axis);
        // fp32->fp16
        float_to_fp16(dout_basic_fp32, dout_basic, tout_basic.numel());
      }
      /// warm up
      for (int i = 0; i < FLAGS_warmup; ++i) {
        ele_add.Launch();
      }
      /// compute
      Timer t0;
      for (int i = 0; i < FLAGS_repeats; ++i) {
        t0.Start();
        ele_add.Launch();
        t0.Stop();
      }

      VLOG(4) << "elementwise_add fp16 x shape: " << x_dim
              << ", y shape: " << y_dim << ", axis: " << axis
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
            int64_t width = x_dim[3];
            print_tensor_info_fp16(basic_ptr, saber_ptr, ptr, size, width);
            LOG(FATAL) << "test fp16 elementwise_add: inputx: " << x_dim
                       << ", inputy: " << y_dim << ", axis: " << axis
                       << ", threads: " << th << ", power_mode: " << cls
                       << " failed!!\n";
          }
        }
      }
      LOG(INFO) << "test fp16 elementwise_add: inputx: " << x_dim
                << ", inputy: " << y_dim << ", axis: " << axis
                << ", threads: " << th << ", power_mode: " << cls
                << " successed!!\n";
    }
  }

  delete param.X;
  delete param.Y;
  delete param.Out;
}

#else
void test_elementwise_fp16(const DDim x_dim,
                           const DDim y_dim,
                           const int axis,
                           const std::vector<int>& thread_num,
                           const std::vector<int>& power_mode) {}
#endif  // LITE_WITH_ARM

#if 1  /// random param pool
TEST(TestEleAddRand, test_ele_add_rand) {
  if (FLAGS_basic_test) {
    for (auto n : {1, 3, 4, 11}) {
      for (auto c : {1, 3, 11, 4}) {
        for (auto h : {1, 3, 11, 4}) {
          for (auto w : {1, 3, 4, 12}) {
            for (auto axis : {0, 1, 2, 3}) {
              DDim x_dim({n, c, h, w});
              DDim y_dim({n, c, h, w});
              test_elementwise_fp16(
                  x_dim, y_dim, axis, {4}, {FLAGS_power_mode});
            }
          }
        }
      }
    }
  }
}
#endif  /// random param conv

#if 1  /// custom
TEST(TesEleAddCustom, test_ele_add_custom_size) {
  test_elementwise_fp16(
      DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width}),
      DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width}),
      FLAGS_axis,
      {FLAGS_threads},
      {FLAGS_power_mode});
}
#endif  // custom
