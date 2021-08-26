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

#include "lite/operators/lrn_op.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

/**
 * @brief get sum of x^2 between channels [size elements]
 *
 * @tparam float
 * @param input
 * @param channel_id: the c-th channel within n-th graph.
 * @param offset_within_channel: the pixel's offset within a channel.
 * @param offset_num: the first address of n-th graph.
 * @param c
 * @param h
 * @param w
 * @param size
 * @return float
 */
float lrn_square(const float* input,
                 int channel_id,
                 int offset_within_channel,
                 int offset_num,
                 int c,
                 int h,
                 int w,
                 int size) {
  int pre_pad = (size - 1) / 2;
  float res = 0;
  const float* src = input + offset_num;

  // handle left channels with padding situation.
  if (channel_id - pre_pad < 0) {
    for (int i = 0; i <= channel_id; ++i) {
      res += src[i * h * w + offset_within_channel] *
             src[i * h * w + offset_within_channel];
    }
  }

  // handle left channels.
  if (channel_id - pre_pad >= 0) {
    for (int i = channel_id - pre_pad; i <= channel_id; ++i) {
      res += src[i * h * w + offset_within_channel] *
             src[i * h * w + offset_within_channel];
    }
  }

  // handle right channels.
  if (channel_id + pre_pad < c) {
    for (int i = channel_id + 1; i <= channel_id + pre_pad; ++i) {
      res += src[i * h * w + offset_within_channel] *
             src[i * h * w + offset_within_channel];
    }
  }

  // handle right channels with padding situation.
  if (channel_id + pre_pad >= c && channel_id + 1 < c) {
    for (int i = channel_id + 1; i < c; ++i) {
      res += src[i * h * w + offset_within_channel] *
             src[i * h * w + offset_within_channel];
    }
  }

  return res;
}

void lrn_compute_ref(std::shared_ptr<operators::LrnOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x =
      scope->FindVar(op_info->Input("X").front())->GetMutable<lite::Tensor>();
  auto out = scope->FindVar(op_info->Output("Out").front())
                 ->GetMutable<lite::Tensor>();

  const float* x_data = x->data<const float>();
  float* out_data = out->mutable_data<float>();
  auto x_dims = x->dims();

  auto alpha = op_info->GetAttr<float>("alpha");
  auto beta = op_info->GetAttr<float>("beta");
  auto k = op_info->GetAttr<float>("k");
  auto norm_region = op_info->GetAttr<std::string>("norm_region");
  auto local_size = op_info->GetAttr<int>("n");

  int N = x_dims[0];
  int C = x_dims[1];
  int H = x_dims[2];
  int W = x_dims[3];

  int offset_num = 0;
  int offset_within_channel = 0;
  int dst_id;

  float square;

  for (int n = 0; n < N; ++n) {
    offset_num = n * C * H * W;

    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
          offset_within_channel = h * W + w;
          dst_id = offset_num + c * H * W + offset_within_channel;
          square = lrn_square(x_data,
                              c,
                              offset_within_channel,
                              offset_num,
                              C,
                              H,
                              W,
                              local_size);
          out_data[dst_id] = x_data[dst_id] * pow(k + alpha * square, -beta);
        }
      }
    }
  }
}

void test_lrn(float alpha,
              float beta,
              float k,
              int local_size,
              int n,
              int c,
              int h,
              int w,
              const std::string& norm_region) {
  Scope scope;
  std::string x_var_name("X_test");
  std::string out_var_name("Out_test");
  std::string out_ref_var_name("Out_ref");
  auto* x = scope.NewTensor(x_var_name);
  auto* out = scope.NewTensor(out_var_name);
  auto* out_ref = scope.NewTensor(out_ref_var_name);

  std::vector<int64_t> x_dim{n, c, h, w};
  x->Resize(x_dim);
  out->Resize(x_dim);
  out_ref->Resize(x_dim);
  auto* x_data = x->mutable_data<float>();
  FillTensor<float, float>(x, 0.f, 1.f);
  float *dmax, *dmin;
  std::tie(dmin, dmax) =
      std::minmax_element(x_data, x_data + x->data_size() - 1);

  cpp::OpDesc opdesc;
  opdesc.SetType("lrn");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("alpha", alpha);
  opdesc.SetAttr("beta", beta);
  opdesc.SetAttr("k", k);
  opdesc.SetAttr("n", local_size);
  opdesc.SetAttr("norm_region", norm_region);
  OpInfo op_info(opdesc);
  op_info.SetInputScale(x_var_name, {(*dmax - *dmin) / 255.f});

  auto op = CreateOp<operators::LrnOpLite>(op_info, &scope);

  // baseline
  lrn_compute_ref(op);
  out_ref->CopyDataFrom(*out);

  Tensor input_x;
  input_x.Resize(x->dims());
  transpose(x->mutable_data<float>(),
            input_x.mutable_data<float>(),
            {static_cast<int>(x_dim[0]),
             static_cast<int>(x_dim[1]),
             static_cast<int>(x_dim[2]),
             static_cast<int>(x_dim[3])},
            {0, 2, 3, 1});
  x->CopyDataFrom(input_x);

  LaunchOp(op, {x_var_name}, {out_var_name});

  Tensor output_trans;
  auto os = out->dims();
  output_trans.Resize(os);
  transpose(out->mutable_data<float>(),
            output_trans.mutable_data<float>(),
            {static_cast<int>(os[0]),
             static_cast<int>(os[2]),
             static_cast<int>(os[3]),
             static_cast<int>(os[1])},
            {0, 3, 1, 2});

  auto output_data = output_trans.mutable_data<float>();
  auto* output_ref_data = out_ref->mutable_data<float>();
  for (size_t i = 0; i < out->data_size(); i++) {
    EXPECT_NEAR(output_data[i], output_ref_data[i], 5e-4);
  }
}

TEST(MLUBridges, lrn) {
  int local_size = 5;
  float alpha = 0.0001f;
  float beta = 0.75;
  float k = 2.0f;
  std::string norm_region = "AcrossChannels";
  for (int w : {2, 4, 8}) {
    for (int h : {2, 4, 8}) {
      for (int c : {1, 2, 3, 4}) {
        for (int n : {1, 2, 3, 4}) {
          test_lrn(alpha, beta, k, local_size, n, c, h, w, norm_region);
        }
      }
    }
  }
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(lrn, kMLU)
