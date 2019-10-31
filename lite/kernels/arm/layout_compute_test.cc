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

#include "lite/kernels/arm/layout_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

#define IN(n, c, h, w)                                 \
  input_data[w + h * input_w + c * input_h * input_w + \
             n * input_c * input_h * input_w]
#define OUT(n, c, h, w)                                    \
  output_data[w + h * output_w + c * output_h * output_w + \
              n * output_c * output_h * output_w]

template <typename Dtype>
void nchw2nhwc_ref(lite::Tensor* input, lite::Tensor* output) {
  auto* input_data = input->data<Dtype>();
  auto* output_data = output->mutable_data<Dtype>();

  int input_n = input->dims()[0];
  int input_c = input->dims()[1];
  int input_h = input->dims()[2];
  int input_w = input->dims()[3];
  int output_c = output->dims()[1];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, h, w, c) = IN(n, c, h, w);
        }
      }
    }
  }
}
#undef IN
#undef OUT

#define IN(n, h, w, c)                                 \
  input_data[c + w * input_c + h * input_w * input_c + \
             n * input_h * input_w * input_c]
#define OUT(n, h, w, c)                                    \
  output_data[c + w * output_c + h * output_w * output_c + \
              n * output_h * output_w * output_c]
template <typename Dtype>
void nhwc2nchw_ref(lite::Tensor* input, lite::Tensor* output) {
  auto* input_data = input->data<Dtype>();
  auto* output_data = output->mutable_data<Dtype>();

  int input_n = input->dims()[0];
  int input_h = input->dims()[1];
  int input_w = input->dims()[2];
  int input_c = input->dims()[3];
  int output_h = output->dims()[1];
  int output_w = output->dims()[2];
  int output_c = output->dims()[3];

  for (int n = 0; n < input_n; ++n) {
    for (int c = 0; c < input_c; ++c) {
      for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
          OUT(n, c, h, w) = IN(n, h, w, c);
        }
      }
    }
  }
}

void print_tensor(lite::Tensor tensor) {
  auto* data = tensor.mutable_data<float>();
  for (int i = 0; i < tensor.numel(); i++) {
    printf("%f ", data[i]);
    if ((i + 1) % 10 == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

TEST(layout_arm, retrive_op) {
  auto layout =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "layout");
  ASSERT_FALSE(layout.empty());
  ASSERT_TRUE(layout.front());
}

TEST(layout_arm, init) {
  NCHWToNHWCCompute layout;
  ASSERT_EQ(layout.precision(), PRECISION(kFloat));
  ASSERT_EQ(layout.target(), TARGET(kARM));
}

TEST(layout_arm, compute) {
  NCHWToNHWCCompute layout;
  operators::LayoutParam param;
  lite::Tensor x;
  lite::Tensor out, out_ref;
  for (auto n : {1, 3}) {
    for (auto c : {1, 3, 5, 32}) {
      for (auto h : {3, 16, 20, 32}) {
        for (auto w : {3, 16, 20, 32}) {
          for (auto nchw2nhwc : {true}) {
            if (nchw2nhwc) {
              x.Resize({n, c, h, w});
              out.Resize({n, h, w, c});
              out_ref.Resize({n, h, w, c});
            } else {
              x.Resize({n, h, w, c});
              out.Resize({n, c, h, w});
              out_ref.Resize({n, c, h, w});
            }
            printf("nchw2nhwc: %d, n: %d. c: %d, h: %d, w: %d \n",
                   nchw2nhwc,
                   n,
                   c,
                   h,
                   w);
            auto* x_data = x.mutable_data<float>();
            auto* out_data = out.mutable_data<float>();
            auto* out_ref_data = out_ref.mutable_data<float>();
            for (int i = 0; i < x.numel(); ++i) {
              x_data[i] = static_cast<float>((i + 1) % 127);
            }
            param.x = &x;
            param.y = &out;
            layout.SetParam(param);
            layout.Run();
            // basic
            if (nchw2nhwc) {
              nchw2nhwc_ref<float>(&x, &out_ref);
            } else {
              nhwc2nchw_ref<float>(&x, &out_ref);
            }
            double diff = 0;
            bool flag = false;
            lite::Tensor vdiff;
            vdiff.Resize({n, c, h, w});
            auto* vdiff_data = vdiff.mutable_data<float>();
            for (int i = 0; i < out.numel(); i++) {
              vdiff_data[i] = out_data[i] - out_ref_data[i];
              if (vdiff_data[i] > 1e-5) {
                flag = true;
                // break;
              }
            }
            if (flag) {
              printf("din: \n");
              print_tensor(x);
              printf("baisc res: \n");
              print_tensor(out_ref);
              printf("lite res: \n");
              print_tensor(out);
              printf("diff: \n");
              print_tensor(vdiff);
              return;
            } else {
              printf("success \n");
            }
          }
        }
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(layout, kARM, kFloat, kNCHW, nchw2nhwc);
