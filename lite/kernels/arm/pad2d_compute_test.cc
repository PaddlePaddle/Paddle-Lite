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

#include "lite/kernels/arm/pad2d_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void pad2d_compute_ref(const lite::Tensor* input,
                       lite::Tensor* output,
                       operators::Pad2dParam param) {
  float* dout = output->mutable_data<float>();
  const float* din = input->data<float>();
  auto output_dims = output->dims();
  // nchw
  int n = output_dims[0];
  int c = output_dims[1];
  int h = output_dims[2];
  int w = output_dims[3];
  int pad_top = param._pad_h[0];
  int pad_bottom = param._pad_h[1];
  int pad_left = param._pad_w[0];
  int pad_right = param._pad_w[1];
  int pad_mode = param._mode;
  float pad_value = param._pad_value;

  int in_w = w - pad_left - pad_right;
  int in_h = h - pad_bottom - pad_top;
  int spatial_size_out = w * h;
  int spatial_size_in = in_w * in_h;
#pragma omp parallel for
  for (int i = 0; i < n * c; ++i) {
    const float* din_batch = din + i * spatial_size_in;
    float* dout_batch = dout + i * spatial_size_out;
    int in_y = 0;
    int in_x = 0;
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        switch (pad_mode) {
          case 0:
            /////////////////////////////
            /*     _mode是PadMode
                   typedef enum{
                       PAD_CONSTANT = 0,
                       PAD_EDGE = 1,
                       PAD_REFLECT = 2,
                   } PadMode;   */
            /////////////////////////
            in_y = y - pad_top;
            in_x = x - pad_left;
            dout_batch[y * w + x] =
                (in_x >= 0 && in_x < in_w) && (in_y >= 0 && in_y < in_h)
                    ? din_batch[in_y * in_w + in_x]
                    : pad_value;
            break;
          case 1:
            in_x =
                std::min(std::max(pad_left, x), in_w + pad_left - 1) - pad_left;
            in_y = std::min(std::max(pad_top, y), in_h + pad_top - 1) - pad_top;
            dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
            break;
          case 2:
            in_y = y - pad_top;
            in_x = x - pad_left;
            in_y = std::max(in_y, -in_y);
            in_y = std::min(in_y, 2 * in_h - in_y - 2);
            in_x = std::max(in_x, -in_x);
            in_x = std::min(in_x, 2 * in_w - in_x - 2);
            dout_batch[y * w + x] = din_batch[in_y * in_w + in_x];
            break;
          default:
            LOG(ERROR) << "ERROR: unknown pad mode:" << pad_mode;
        }
      }
    }
  }
}

TEST(pad2d_arm, retrive_op) {
  auto pad2d =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>("pad2d");
  ASSERT_FALSE(pad2d.empty());
  ASSERT_TRUE(pad2d.front());
}

TEST(pad2d_arm, init) {
  Pad2dCompute pad2d;
  ASSERT_EQ(pad2d.precision(), PRECISION(kFloat));
  ASSERT_EQ(pad2d.target(), TARGET(kARM));
}

TEST(pad2d_arm, compute) {
  // 1、原始变量/////////
  Pad2dCompute pad2d;
  operators::Pad2dParam param;
  lite::Tensor tensorA;
  lite::Tensor output;
  lite::Tensor output_ref;

  for (int pad_top : {0, 1}) {
    for (int pad_bottom : {0, 1}) {
      std::vector<int> pad_h{pad_top, pad_bottom};
      for (int pad_left : {0, 1}) {
        for (int pad_right : {0, 1}) {
          std::vector<int> pad_w{pad_left, pad_right};
          for (int pad_mode : {0, 1, 2}) {
            for (float pad_value : {0.f, 1.0f}) {
              param._pad_h = pad_h;
              param._pad_w = pad_w;
              param._pad_value = pad_value;
              param._mode = pad_mode;
              // Pad2DParam<TargetType_D> param(pad_h, pad_w, pad_value, _mode);
              LOG(INFO) << "pad param: " << pad_mode << " " << pad_value << " "
                        << pad_h[0] << " " << pad_h[1] << " " << pad_w[0] << " "
                        << pad_w[1];
              for (int n : {1, 2}) {
                for (int c : {1, 3}) {
                  for (int h : {14, 24}) {
                    for (int w : {14, 24}) {
                      DDimLite ddimA({n, c, h, w});
                      tensorA.Resize(ddimA);
                      output.Resize(ddimA);
                      output_ref.Resize(ddimA);
                      auto* output_data = output.mutable_data<float>();
                      auto* output_ref_data = output_ref.mutable_data<float>();
                      for (int i = 0; i < ddimA.production(); ++i) {
                        output_data[i] = -2;
                        output_ref_data[i] = -2;
                      }
                      for (int i = 0; i < ddimA.data()[0] * ddimA.data()[1] *
                                              ddimA.data()[2] * ddimA.data()[3];
                           i++) {
                        tensorA.mutable_data<float>()[i] = i;
                      }
                      param.X = &tensorA;
                      param.Out = &output;
                      pad2d.SetParam(param);
                      LOG(INFO) << "test pad2d start :";
                      pad2d.Run();
                      LOG(INFO) << "pad2d.Run end";
                      param.Out = &output_ref;
                      LOG(INFO) << "concat_compute_ref start";
                      pad2d_compute_ref(param.X, &output_ref, param);
                      LOG(INFO) << "concat_compute_ref end";
                      int dim_out = output.dims().production();

                      output_data = output.mutable_data<float>();
                      output_ref_data = output_ref.mutable_data<float>();
                      for (int i = 0; i < dim_out; i++) {
                        EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
                      }
                    }
                  }
                }
              }
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

USE_LITE_KERNEL(pad2d, kARM, kFloat, kNCHW, def);
