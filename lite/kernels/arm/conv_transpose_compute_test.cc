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

#include "lite/kernels/arm/conv_transpose_compute.h"
#include <gtest/gtest.h>
#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>
#include <vector>
#include "lite/arm/math/funcs.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename type, typename type2>
static void basic_gemm(int m,
                       int n,
                       int k,
                       const type* a,
                       const type* b,
                       const type2* bias,
                       type2* c,
                       type2 alpha,
                       type2 beta,
                       bool trans_a = false,
                       bool trans_b = false,
                       bool flag_bias = false,
                       bool flag_relu = false) {
#pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    type2 bias_data = (type2)0;
    if (flag_bias) {
      bias_data = bias[i];
    }
    for (int j = 0; j < n; ++j) {
      type2 sum = static_cast<type2>(0);
      for (int l = 0; l < k; ++l) {
        type av;
        type bv;
        if (trans_a) {
          av = a[l * m + i];
        } else {
          av = a[i * k + l];
        }
        if (trans_b) {
          bv = b[j * k + l];
        } else {
          bv = b[l * n + j];
        }
        sum += av * bv;
      }
      type2 tmp = alpha * sum + beta * c[i * n + j] + bias_data;
      if (flag_relu) {
        c[i * n + j] = tmp > (type2)0 ? tmp : (type2)0;
      } else {
        c[i * n + j] = tmp;
      }
    }
  }
}

//! for float, dtype1 and type2 is float
//! for int8, dytpe1 is char, dtype2 is int
template <typename Dtype1, typename Dtype2>
bool deconv_basic(const Dtype1* din,
                  Dtype2* dout,
                  int num,
                  int chout,
                  int hout,
                  int wout,
                  int chin,
                  int hin,
                  int win,
                  const Dtype1* weights,
                  const Dtype2* bias,
                  int group,
                  int kernel_w,
                  int kernel_h,
                  int stride_w,
                  int stride_h,
                  int dila_w,
                  int dila_h,
                  int pad_w,
                  int pad_h,
                  bool flag_bias,
                  bool flag_relu) {
  int m = chout * kernel_w * kernel_h / group;
  int n = hin * win;
  int k = chin / group;

  if (chin != chout || group != chin) {
    CHECK_OR_FALSE(chin % group == 0);
    CHECK_OR_FALSE(chout % group == 0);
  }

  lite::Tensor workspace_tensor;
  std::vector<int64_t> wt_shape = {1, 1, 1, group * m * n};
  workspace_tensor.Resize(wt_shape);
  auto* workspace_ptr = workspace_tensor.mutable_data<Dtype2>();

  int group_size_in = win * hin * chin / group;
  int group_size_out = wout * hout * chout / group;
  int group_size_coldata = m * n;
  int group_size_weights = chin * chout * kernel_w * kernel_h / (group * group);
  bool flag_1x1s1p1 = (kernel_w == 1) && (kernel_h == 1) && (stride_h == 1) &&
                      (stride_w == 1) && (pad_w == 1) && (pad_h == 1) &&
                      (dila_w == 1) && (dila_h == 1);

  for (int i = 0; i < num; ++i) {
    const Dtype1* din_batch = din + i * chin * hin * win;
    Dtype2* dout_batch = dout + i * chout * hout * wout;

    Dtype2* col_data = workspace_ptr;
    if (flag_1x1s1p1) {
      col_data = dout_batch;
    }
    memset(col_data, 0, sizeof(Dtype2) * group_size_coldata);
    for (int g = 0; g < group; ++g) {
      const Dtype1* din_group = din_batch + g * group_size_in;
      const Dtype1* weights_group = weights + g * group_size_weights;
      Dtype2* coldata_group = col_data + g * group_size_coldata;
      basic_gemm<Dtype1, Dtype2>(m,
                                 n,
                                 k,
                                 weights_group,
                                 din_group,
                                 nullptr,
                                 coldata_group,
                                 (Dtype2)1,
                                 (Dtype2)0,
                                 true,
                                 false,
                                 false,
                                 (!flag_bias && flag_relu));
    }
    if (!flag_1x1s1p1) {
      lite::arm::math::col2im(col_data,
                              chout,
                              hout,
                              wout,
                              kernel_h,
                              kernel_w,
                              pad_h,
                              pad_w,
                              stride_h,
                              stride_w,
                              dila_h,
                              dila_w,
                              dout_batch);
    }
    if (flag_bias) {
      lite::arm::math::fill_bias_relu(
          dout_batch, bias, chout, wout * hout, flag_bias, flag_relu);
    }
  }
  return true;
}

template <typename dtype>
void conv2d_transpose_compute_ref(const operators::ConvParam& param) {}

TEST(conv2d_transpose_arm, retrive_op) {
  auto op = KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
      "conv2d_transpose");
  ASSERT_FALSE(op.empty());
  ASSERT_TRUE(op.front());
}

TEST(conv2d_transpose_arm, init) {
  Conv2DTransposeCompute compute;
  ASSERT_EQ(compute.precision(), PRECISION(kFloat));
  ASSERT_EQ(compute.target(), TARGET(kARM));
}
TEST(conv2d_transpose_arm, compute) {
  /*
    DeviceInfo::Init();
    for (auto n : {2, 3}) {
      for (auto c : {3, 4 }) {
        for (auto h : {4, 5}) {
          for (auto w : {5, 6}) {
            Tensor x;
            Tensor output;
            Tensor output_ref;
            int axis = (n + c + h + w) % 4;

            // get tensor x data
            x.Resize({n, c, h, w});
            auto* x_data = x.mutable_data<float>();
            for (int i = 0; i < x.dims().production(); i++) {
              float sign = i % 3 == 0 ? -1.0f : 1.0f;
              x_data[i] = sign * static_cast<float>(i % 128) * 0.013f;
            }

            // resize output and output_ref
            int nchw[] = {n, c, h, w};
            std::vector<int64_t> output_size(nchw, nchw + 4);
            output_size.erase(output_size.begin() + axis);
            output.Resize(output_size);
            output_ref.Resize(output_size);

            // obtain output_data
            ArgmaxCompute argmaxOp;
            std::unique_ptr<KernelContext> ctx(new KernelContext);
            ctx->As<ARMContext>();
            argmaxOp.SetContext(std::move(ctx));
            operators::ArgmaxParam param;
            param.x = &x;
            param.output = &output;
            param.axis = axis;
            argmaxOp.SetParam(param);
            argmaxOp.Launch();
            auto* output_data = output.mutable_data<float>();

            // obtain output_ref_data
            param.output = &output_ref;
            argmax_compute_ref<float>(param);
            auto* output_ref_data = output_ref.mutable_data<float>();

            // compare
            for (int i = 0; i < output.dims().production(); i++) {
              EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
            }
          }
        }
      }
    }
  */
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
