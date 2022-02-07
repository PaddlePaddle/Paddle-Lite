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

#include "lite/kernels/x86/fc_compute.h"
#include "lite/backends/x86/math/gemm_s8u8_compute.h"
#include "lite/backends/x86/math/saturate.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

#define GEMM_OUT_INT8                                                     \
  lite::x86::math::generate_gemm_s8u8_x86_kern<int8_t> gemm(false,        \
                                                            false,        \
                                                            m,            \
                                                            n,            \
                                                            k,            \
                                                            i_data,       \
                                                            n,            \
                                                            w_scale,      \
                                                            input_scale,  \
                                                            output_scale, \
                                                            b_data,       \
                                                            relu_type,    \
                                                            1.f);

#define GEMM_OUT_FLOAT                                                   \
  lite::x86::math::generate_gemm_s8u8_x86_kern<float> gemm(false,        \
                                                           false,        \
                                                           m,            \
                                                           n,            \
                                                           k,            \
                                                           i_data,       \
                                                           n,            \
                                                           w_scale,      \
                                                           input_scale,  \
                                                           output_scale, \
                                                           b_data,       \
                                                           relu_type,    \
                                                           1.f);

template <lite::TargetType Target, typename T>
class FCFunctor {
 public:
  void operator()(const lite::X86Context& context,
                  const int M,
                  const int N,
                  const int K,
                  const T* X,
                  const T* W,
                  T* Y,
                  const T* B = nullptr,
                  bool relu = false,
                  bool padding_weights = false) {
    auto blas = lite::x86::math::GetBlas<lite::TargetType::kX86, T>(context);
    T* Y1_data = nullptr;

    auto compute =
        relu
            ? jit::KernelFuncs<jit::VAddReluTuple<T>, fluid::CPUPlace>::Cache()
                  .At(N)
            : jit::KernelFuncs<jit::VAddTuple<T>, fluid::CPUPlace>::Cache().At(
                  N);
    auto parallel_compute = [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; i++) {
        T* dst = Y + i * N;
        T* src = Y1_data ? Y1_data + i * (N + 4) : dst;
        compute(B, src, dst, N);
      }
    };

    // Because of the overhead of memcpy, we only do padding for GEMM
    //  when weights is already padded in fc_fuse_pass.
    if (padding_weights) {
      const int NN = N + 4;
      const int KK = K + 4;

      // NOTE: here need to mutable_data for temporary Tensor X1 and Y1,
      //  the overhead is unmeasured.
      lite::Tensor X1;
      X1.Resize(std::vector<int64_t>{M * KK});
      T* X1_data = X1.mutable_data<T>();

      lite::Tensor Y1;
      Y1.Resize(std::vector<int64_t>{M * NN});
      Y1_data = Y1.mutable_data<T>();

      auto parallel_memcpy_x = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; i++) {
          memcpy(X1_data + i * KK, X + i * K, K * sizeof(T));
        }
      };

      parallel_memcpy_x(0, M);

      blas.GEMM(false,
                false,
                M,
                N,
                K,
                static_cast<T>(1.0),
                X1_data,
                KK,
                W,
                NN,
                static_cast<T>(0.0),
                Y1_data,
                NN);

      if (!B) {
        auto parallel_memcpy_y = [&](int64_t begin, int64_t end) {
          for (int64_t i = begin; i < end; i++) {
            memcpy(Y + i * N, Y1_data + i * NN, N * sizeof(T));
          }
        };
        parallel_memcpy_y(0, M);
        return;
      }
      parallel_compute(0, M);
    } else {
      blas.MatMul(M, N, K, X, W, Y);
      if (!B) {
        return;
      }
      parallel_compute(0, M);
    }
  }
};

template <>
void FcCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = *param_.get_mutable<param_t>();
  auto* input = param.input;
  auto* w = param.w;
  auto* bias = param.bias;
  auto* output = param.output;
  bool with_relu = (param.activation_type == "relu") ? true : false;

  bool padding_weights = param.padding_weights;
  const auto& w_dims = w->dims();
  auto w_dims0 = padding_weights ? w_dims[0] - 4 : w_dims[0];
  auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];

  int M = output->dims().production() / w_dims1;

  const float* input_data = input->template data<float>();
  const float* w_data = w->template data<float>();
  float* output_data = output->template mutable_data<float>();

  auto& context = ctx_->As<X86Context>();
  FCFunctor<lite::TargetType::kX86, float> fc;
  fc(context,
     M,
     w_dims1,
     w_dims0,
     input_data,
     w_data,
     output_data,
     bias ? bias->template data<float>() : NULL,
     with_relu,
     padding_weights);
}

template <>
void FcCompute<PRECISION(kInt8), PRECISION(kInt8)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto* i_data = param.input->data<int8_t>();
  auto* o_data = param.output->mutable_data<int8_t>();
  auto* w_data = param.w->data<int8_t>();
  const float* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto w_dims = param.w->dims();
  int k = w_dims[0];
  int n = w_dims[1];
  int m = param.output->dims().production() / n;
  float input_scale = param.input_scale;
  float output_scale = param.output_scale;
  int relu_type = (param.activation_type == "relu") ? 1 : 0;
  float* w_scale =
      static_cast<float*>(TargetMalloc(TARGET(kX86), m * sizeof(float)));

  if (param.activation_type != "" && param.activation_type != "relu")
    LOG(FATAL) << "not support fuse activation except relu.";

  if (param.weight_scale.size() == 1) {
    for (int i = 0; i < m; i++) w_scale[i] = param.weight_scale[0];
    GEMM_OUT_INT8;
    gemm.compute(i_data, w_data, o_data);
  } else if (param.weight_scale.size() == m) {
    for (int i = 0; i < m; i++) w_scale[i] = param.weight_scale[i];
    GEMM_OUT_INT8;
    gemm.compute(i_data, w_data, o_data);
  } else if (param.weight_scale.size() == n) {
    for (int i = 0; i < m; i++) w_scale[i] = 1.f;
    float* tmp_output =
        static_cast<float*>(TargetMalloc(TARGET(kX86), m * n * sizeof(float)));
    GEMM_OUT_FLOAT;
    gemm.compute(i_data, w_data, tmp_output);
    for (int nn = 0; nn < n; nn++) {
      float tmp_scale = param.weight_scale[nn] / output_scale;
      for (int mm = 0; mm < m; mm++) {
        int offt = mm * n + nn;
        o_data[offt] = lite::x86::math::saturate_cast<int8_t>(
            roundf(tmp_output[offt] * tmp_scale));
        o_data[offt] = o_data[offt] < -127 ? -127 : o_data[offt];
      }
    }
    TargetFree(TARGET(kX86), tmp_output);
  } else {
    LOG(FATAL) << "weight scale size is not 1, N or M, not support yet.";
  }
  TargetFree(TARGET(kX86), w_scale);
}

template <>
void FcCompute<PRECISION(kInt8), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<operators::FcParam>();
  auto* i_data = param.input->data<int8_t>();
  auto* o_data = param.output->mutable_data<float>();
  auto* w_data = param.w->data<int8_t>();
  const float* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto w_dims = param.w->dims();
  int k = w_dims[0];
  int n = w_dims[1];
  int m = param.output->dims().production() / n;
  int relu_type = (param.activation_type == "relu") ? 1 : 0;
  float input_scale = param.input_scale;
  float output_scale = param.output_scale;
  float* w_scale =
      static_cast<float*>(TargetMalloc(TARGET(kX86), m * sizeof(float)));

  if (param.activation_type != "" && param.activation_type != "relu")
    LOG(FATAL) << "not support fuse activation except relu.";

  if (param.weight_scale.size() == 1) {
    for (int i = 0; i < m; i++) w_scale[i] = param.weight_scale[0];
    GEMM_OUT_FLOAT;
    gemm.compute(i_data, w_data, o_data);
  } else if (param.weight_scale.size() == m) {
    for (int i = 0; i < m; i++) w_scale[i] = param.weight_scale[i];
    GEMM_OUT_FLOAT;
    gemm.compute(i_data, w_data, o_data);
  } else if (param.weight_scale.size() == n) {
    for (int i = 0; i < m; i++) w_scale[i] = 1.f;
    GEMM_OUT_FLOAT;
    gemm.compute(i_data, w_data, o_data);
    for (int nn = 0; nn < n; nn++) {
      float tmp_scale = param.weight_scale[nn];
      for (int mm = 0; mm < m; mm++) {
        int offt = mm * n + nn;
        o_data[offt] = o_data[offt] * tmp_scale;
      }
    }
  } else {
    LOG(FATAL) << "weight scale size is not 1, N or M, not support yet.";
  }
  TargetFree(TARGET(kX86), w_scale);
}

#undef GEMM_OUT_INT8
#undef GEMM_OUT_FLOAT

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::x86::FcCompute<PRECISION(kFloat),
                                              PRECISION(kFloat)>
    FcCompute_FP32;
typedef paddle::lite::kernels::x86::FcCompute<PRECISION(kInt8),
                                              PRECISION(kFloat)>
    FcCompute_int8_fp32;
typedef paddle::lite::kernels::x86::FcCompute<PRECISION(kInt8),
                                              PRECISION(kInt8)>
    FcCompute_int8_int8;

REGISTER_LITE_KERNEL(fc, kX86, kFloat, kNCHW, FcCompute_FP32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();

REGISTER_LITE_KERNEL(fc, kX86, kInt8, kNCHW, FcCompute_int8_int8, int8_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .Finalize();

REGISTER_LITE_KERNEL(fc, kX86, kInt8, kNCHW, FcCompute_int8_fp32, fp32_out)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kX86))})
    .BindInput("W", {LiteType::GetTensorTy(TARGET(kX86), PRECISION(kInt8))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kX86))})
    .Finalize();
