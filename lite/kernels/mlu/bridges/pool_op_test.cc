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

#include "lite/operators/pool_op.h"
#include <gtest/gtest.h>
#include <random>
#include "lite/core/op_registry.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void pool_ref(const std::shared_ptr<operators::PoolOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto& in_dims = x->dims();
  auto& out_dims = out->dims();

  const float* src_ptr = x->data<const float>();
  float* dst_ptr = out->mutable_data<float>();

  std::vector<int> ksize = op_info->GetAttr<std::vector<int>>("ksize");
  std::vector<int> strides = op_info->GetAttr<std::vector<int>>("strides");
  std::vector<int> paddings = op_info->GetAttr<std::vector<int>>("paddings");
  bool exclusive = op_info->GetAttr<bool>("exclusive");
  std::string pooling_type = op_info->GetAttr<std::string>("pooling_type");
  bool global_pooling = op_info->GetAttr<bool>("global_pooling");

  if (pooling_type == "max") {
    for (int i = 0; i < out_dims.production(); ++i) {
      dst_ptr[i] = -65504.f;
    }
  }

  int in_n = in_dims[0];
  int in_c = in_dims[1];
  int in_h = in_dims[2];
  int in_w = in_dims[3];
  int size_in_n = in_c * in_h * in_w;
  int size_in_c = in_h * in_w;

  int out_h = out_dims[2];
  int out_w = out_dims[3];
  int size_out_n = in_c * out_h * out_w;
  int size_out_c = out_h * out_w;

  int window_h = ksize[0];
  int window_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[2];

  if (global_pooling == true) {
    for (int n = 0; n < in_n; ++n) {
      for (int c = 0; c < in_c; ++c) {
        const float* src = src_ptr + n * size_in_n + c * size_in_c;
        float res = src[0];
        if (pooling_type == "max") {
          for (int i = 1; i < size_in_c; ++i) {
            float cur_val = src[i];
            res = cur_val > res ? cur_val : res;
          }
        } else if (pooling_type == "avg") {
          for (int i = 1; i < size_in_c; ++i) {
            float cur_val = src[i];
            res += cur_val;
          }
          res /= size_in_c;
        }
        dst_ptr[n * size_out_n + c] = res;
      }
    }
  } else {
    for (int n = 0; n < in_n; ++n) {
      for (int c = 0; c < in_c; ++c) {
        for (int h = 0; h < out_h; ++h) {
          int sh = h * stride_h;
          int eh = sh + window_h;
          sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
          eh = (eh - pad_h) > in_h ? in_h : eh - pad_h;
          for (int w = 0; w < out_w; ++w) {
            int sw = w * stride_w;
            int ew = sw + window_w;
            sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
            ew = (ew - pad_w) > in_w ? in_w : ew - pad_w;
            int pooling_size = (ew - sw) * (eh - sh);
            if (pooling_size == 0) continue;
            float res = 0.f;
            for (int kh = sh; kh < eh; ++kh) {
              for (int kw = sw; kw < ew; ++kw) {
                int src_idx = n * size_in_n + c * size_in_c + kh * in_w + kw;
                if (kh == sh && kw == sw) {
                  res = src_ptr[src_idx];
                } else {
                  if (pooling_type == "max") {
                    res = res >= src_ptr[src_idx] ? res : src_ptr[src_idx];
                  }
                  if (pooling_type == "avg") {
                    res += src_ptr[src_idx];
                  }
                }
              }
            }
            if (pooling_type == "avg") {
              if (exclusive) {
                res /= pooling_size;
              } else {
                res /= window_h * window_w;
              }
            }
            dst_ptr[n * size_out_n + c * size_out_c + h * out_w + w] = res;
          }
        }
      }
    }
  }
}

void test_pool(int bs,
               int ic,
               int ih,
               int iw,
               std::string pooling_type,
               bool ceil_mode,
               bool global_pooling,
               bool exclusive,
               int ksize,
               int stride,
               int padding) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});

  // initialize input&output data
  FillTensor<float>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("pool2d");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("pooling_type", pooling_type);
  opdesc.SetAttr("ksize", std::vector<int>({ksize, ksize}));
  opdesc.SetAttr("global_pooling", global_pooling);
  opdesc.SetAttr("exclusive", exclusive);
  opdesc.SetAttr("ceil_mode", ceil_mode);
  opdesc.SetAttr("strides", std::vector<int>({stride, stride}));
  opdesc.SetAttr("paddings",
                 std::vector<int>({padding, padding, padding, padding}));

  // create and convert op to MLU model, then run it on MLU
  auto op = CreateOp<operators::PoolOpLite>(opdesc, &scope);
  // execute reference implementation and save to output tensor
  pool_ref(op);
  out_ref->CopyDataFrom(*out);

  Tensor input_trans;
  input_trans.Resize({bs, ic, ih, iw});
  transpose(x->mutable_data<float>(),
            input_trans.mutable_data<float>(),
            {bs, ic, ih, iw},
            {0, 2, 3, 1});

  auto os = out->dims();
  x->CopyDataFrom(input_trans);

  LaunchOp(op, {x_var_name}, {out_var_name});

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  Tensor output_trans;
  output_trans.Resize(out->dims());
  transpose(out_data,
            output_trans.mutable_data<float>(),
            {static_cast<int>(os[0]),
             static_cast<int>(os[2]),
             static_cast<int>(os[3]),
             static_cast<int>(os[1])},
            {0, 3, 1, 2});
  out_data = output_trans.mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(MLUBridges, pool) {
  for (auto pooling_type : {"max", "avg"}) {
    for (auto ceil_mode : {true, false}) {
      for (auto global_pooling : {true, false}) {
        for (auto exclusive : {true /*, false*/}) {
          for (auto ksize : {2, 3}) {
            for (auto stride : {1, 2}) {
              for (auto padding : {0, 1}) {
                for (auto bs : {1, 3}) {
                  for (auto ic : {1, 3}) {
                    for (auto ih : {3, 7}) {
                      for (auto iw : {3, 7}) {
                        LOG(INFO)
                            << "shape: " << bs << ',' << ic << ',' << ih << ','
                            << iw << '\t' << "pooling type: " << pooling_type
                            << '\t' << "ceil model: " << ceil_mode << '\t'
                            << "global_pooling: " << global_pooling << '\t'
                            << "exclusive: " << exclusive << '\t'
                            << "ksize: " << ksize << '\t'
                            << "stride: " << stride << '\t'
                            << "padding: " << padding;
                        test_pool(bs,
                                  ic,
                                  ih,
                                  iw,
                                  pooling_type,
                                  ceil_mode,
                                  global_pooling,
                                  exclusive,
                                  ksize,
                                  stride,
                                  padding);
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

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(pool2d, kMLU)
