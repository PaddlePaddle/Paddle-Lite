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

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include "lite/core/mir/pattern_matcher_high_api.h"
#include "lite/utils/cp_logging.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class ConvConvFuser : public FuseBase {
 public:
  explicit ConvConvFuser(const std::string& conv_type0,
                         const std::string& conv_type1,
                         const bool conv_has_bias0,
                         const bool conv_has_bias1)
                         : conv_type0_(conv_type0),
                           conv_type1_(conv_type1),
                           conv_has_bias0_(conv_has_bias0),
                           conv_has_bias1_(conv_has_bias1){}
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 private:
  void ComputeNewWeight(float* dout, const float* din, const float* weights,
                        int oc0, int ic, int ih, int iw, int oc1) {
    // input conv_weight0_t weights conv_weight1_t
    // output weight_tensor
    // ksize = 1
    int in_size = ih * iw;
    int in_channel_size = ic * in_size;
    // out = w1[j, i, ih, iw] * w2[k, j, kw, kh]
    // out_dim = [oc1, ic, kh, kw], din_dim = [oc0, ic, kh, kw], weight_dim = [oc1, oc0, kh, kw]
    for (int k = 0; k < oc1; k ++) {
      const float* weights_ptr = weights + k * oc0;
      float* out_ptr = dout + k * in_channel_size;
      for (int c = 0; c < ic; c++) {
        float* out_ptr_channel = out_ptr + c * in_size;
        const float* din_ptr = din + c * in_size;
        for (int i = 0; i < in_size; i++) {
          float sum = 0.f;
          for (int j = 0; j < oc0; j++) {
            sum += din_ptr[j * in_channel_size] * weights_ptr[j];
          }
          *out_ptr_channel++ = sum;
        }
      }
    }
    LOG(INFO) << "din: ";
    for (int i = 0; i < oc0 * in_channel_size; i++){
      if ((i + 1) % oc0 == 0) printf("\n");
      printf("%f  ", din[i]);
    }
    printf("\n");
    LOG(INFO) << "weights: ";
    for (int i = 0; i < oc1 * oc0 * in_size; i++){
      if ((i + 1) % oc1 == 0) printf("\n");
      printf("%f  ", weights[i]);
    }
    printf("\n");
    LOG(INFO) << "out: ";
    for (int i = 0; i < oc1 * in_channel_size; i++){
      if ((i + 1) % oc1 == 0) printf("\n");
      printf("%f  ", dout[i]);
    }
    printf("\n");
  }

void ComputeNewBias(float* dout, Tensor* bias0_tensor, Tensor* weight_tensor, Tensor* bias1_tensor) {
    // input bias0_tensor weight_tensor bias1_tensor
    // output bias_tensor
    auto in_dims = bias0_tensor->dims();
    auto weight_dims = weight_tensor->dims();
    const float* din = bias0_tensor->data<float>();
    const float* weights = weight_tensor->data<float>();
    int ic = in_dims[0];
    int oc = weight_dims[0];
    LOG(INFO) << "in_dims size: " << in_dims.size();
    LOG(INFO) << "weight_dims: " << weight_dims[0] << ", " << weight_dims[1] << ", " << weight_dims[2] << ". " << weight_dims[3];
    float* tmp = dout;
    // out_k = b0[num, j, 1, 1] * w2[k, j, 1, 1]
    if (bias1_tensor) {
      const float* din2 = bias1_tensor->data<float>();
      for (int k = 0; k < oc; k++) {
        const float* weights_ptr = weights + k * ic;
        float sum = 0.f;
        for (int j = 0; j < ic; j++) {
          sum += *din++ * *weights_ptr++;
        }
        *dout++ = sum + *din2;
        din2++;
      }
    } else {
      for (int k = 0; k < oc; k++) {
        const float* weights_ptr = weights + k * ic;
        float sum = 0.f;
        for (int j = 0; j < ic; j++) {
          sum += *din++ * *weights_ptr++;
        }
        *dout++ = sum;
      }
    }
    LOG(INFO) << "din: ";
    for (int i = 0; i < ic; i++){
      printf("%f  ", din[i]);
    }
    printf("\n");
    LOG(INFO) << "weights: ";
    for (int i = 0; i < oc * ic; i++){
      if ((i + 1) % oc == 0) printf("\n");
      printf("%f  ", weights[i]);
    }
    printf("\n");
    if (bias1_tensor) {
      const float* din2 = bias1_tensor->data<float>();
      LOG(INFO) << "din2: ";
      for (int i = 0; i < oc; i++){
        printf("%f  ", din2[i]);
      }
      printf("\n");
    }
    LOG(INFO) << "out: ";
    for (int i = 0; i < oc; i++){
      printf("%f  ", tmp[i]);
    }
    printf("\n");
  }

 private:
  std::string conv_type0_{"conv2d"};
  std::string conv_type1_{"conv2d"};
  bool conv_has_bias0_{false};
  bool conv_has_bias1_{false};
};

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
