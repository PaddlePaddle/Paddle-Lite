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

#include "lite/operators/reduce_mean_op.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include "lite/core/op_registry.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/npu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {
namespace bridges {

void reduce_mean_n(const float* src,
                   float* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  int hw_size = height_in * width_in;
  int chw_size = channel_in * hw_size;
  int data_index, src_index;
  for (int c = 0; c < channel_in; ++c) {
    for (int h = 0; h < height_in; ++h) {
      for (int w = 0; w < width_in; ++w) {
        data_index = c * hw_size + h * width_in + w;
        dst[data_index] = 0.0;
        for (int n = 0; n < num_in; ++n) {
          src_index = n * chw_size + data_index;
          dst[data_index] += static_cast<float>(src[src_index]) / num_in;
        }
      }
    }
  }
}

void reduce_mean_c(const float* src,
                   float* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  int hw_size = height_in * width_in;
  int chw_size = hw_size * channel_in;
  int data_index, src_index0, src_index;
  for (int n = 0; n < num_in; ++n) {
    for (int h = 0; h < height_in; ++h) {
      for (int w = 0; w < width_in; ++w) {
        data_index = n * hw_size + h * width_in + w;
        src_index0 = n * chw_size + h * width_in + w;
        dst[data_index] = 0.0;
        for (int c = 0; c < channel_in; ++c) {
          src_index = src_index0 + c * hw_size;
          dst[data_index] += static_cast<float>(src[src_index]) / channel_in;
        }
      }
    }
  }
}

void reduce_mean_h(const float* src,
                   float* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  int cw_size = channel_in * width_in;
  int chw_size = cw_size * height_in;
  int hw_size = height_in * width_in;
  int data_index, src_index, src_index0;
  for (int n = 0; n < num_in; ++n) {
    for (int c = 0; c < channel_in; ++c) {
      for (int w = 0; w < width_in; ++w) {
        data_index = n * cw_size + c * width_in + w;
        src_index0 = n * chw_size + c * hw_size + w;
        dst[data_index] = 0.0;
        for (int h = 0; h < height_in; ++h) {
          src_index = src_index0 + h * width_in;
          dst[data_index] += static_cast<float>(src[src_index]) / height_in;
        }
      }
    }
  }
}

void reduce_mean_w(const float* src,
                   float* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  int ch_size = channel_in * height_in;
  int hw_size = height_in * width_in;
  int chw_size = ch_size * width_in;
  int data_index = 0;
  int src_index0 = 0;
  int src_index = 0;
  for (int n = 0; n < num_in; ++n) {
    for (int c = 0; c < channel_in; ++c) {
      for (int h = 0; h < height_in; ++h) {
        data_index = n * ch_size + c * height_in + h;
        src_index0 = n * chw_size + c * hw_size + h * width_in;
        dst[data_index] = 0.0;
        for (int w = 0; w < width_in; ++w) {
          src_index = src_index0 + w;
          dst[data_index] += static_cast<float>(src[src_index]) / width_in;
        }
      }
    }
  }
}

void reduce_mean_all(const float* src,
                     float* dst,
                     int num_in,
                     int channel_in,
                     int height_in,
                     int width_in) {
  float mean = 0.0;
  int src_index;
  int n_id, c_id;
  int all = num_in * channel_in * height_in * width_in;
  for (int n = 0; n < num_in; ++n) {
    n_id = n * channel_in * height_in * width_in;
    for (int c = 0; c < channel_in; ++c) {
      c_id = c * height_in * width_in;
      for (int h = 0; h < height_in; ++h) {
        for (int w = 0; w < width_in; ++w) {
          src_index = n_id + c_id + h * width_in + w;
          mean = src[src_index] / all;
        }
      }
    }
  }
  dst[0] = mean;
}

void reduce_mean_nc(const float* src,
                    float* dst,
                    int num_in,
                    int channel_in,
                    int height_in,
                    int width_in) {
  // reduce n first.
  DDimLite ddimA({1, channel_in, height_in, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  float* tmp_out = tensor_tmp.mutable_data<float>();
  reduce_mean_n(src, tmp_out, num_in, channel_in, height_in, width_in);
  reduce_mean_c(tmp_out, dst, 1, channel_in, height_in, width_in);
}

void reduce_mean_ch(const float* src,
                    float* dst,
                    int num_in,
                    int channel_in,
                    int height_in,
                    int width_in) {
  // reduce c first
  DDimLite ddimA({num_in, 1, height_in, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  float* tmp_out = tensor_tmp.mutable_data<float>();
  reduce_mean_c(src, tmp_out, num_in, channel_in, height_in, width_in);
  reduce_mean_h(tmp_out, dst, num_in, 1, height_in, width_in);
}

void reduce_mean_hw(const float* src,
                    float* dst,
                    int num_in,
                    int channel_in,
                    int height_in,
                    int width_in) {
  // reduce h first
  DDimLite ddimA({num_in, channel_in, 1, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  float* tmp_out = tensor_tmp.mutable_data<float>();
  reduce_mean_h(src, tmp_out, num_in, channel_in, height_in, width_in);
  reduce_mean_w(tmp_out, dst, num_in, channel_in, 1, width_in);
}

void reduce_mean_ref(const std::shared_ptr<operators::ReduceMeanOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();

  auto x = scope->FindTensor("x");
  auto x_dims = x->dims();
  auto x_data = x->data<float>();
  auto out = scope->FindMutableTensor("out_ref");

  auto dim = op_info->GetAttr<std::vector<int>>("dim");
  auto keep_dim = op_info->GetAttr<bool>("keep_dim");

  auto x_rank = x_dims.size();
  if (!dim.empty()) {
    for (size_t i = 0; i < dim.size(); i++) {
      if (dim[i] < 0) {
        dim[i] += x_rank;
      }
    }
  }

  bool reduce_all = false;
  std::stable_sort(dim.begin(), dim.end());
  if (dim.size() == 0) {
    reduce_all = true;
  }

  std::vector<int64_t> out_dims;
  if (reduce_all) {
    if (keep_dim) {
      for (size_t i = 0; i < x_dims.size(); i++) {
        out_dims.push_back(1);
      }
    } else {
      out_dims.push_back(1);
    }
  } else {
    for (int i = 0; i < x_dims.size(); i++) {
      out_dims.push_back(x_dims[i]);
    }
    if (keep_dim) {
      for (size_t i = 0; i < dim.size(); ++i) {
        out_dims[dim[i]] = 1L;
      }
    } else {
      int64_t kDelFlag = -2;
      for (size_t i = 0; i < dim.size(); ++i) {
        out_dims[dim[i]] = kDelFlag;
      }
      out_dims.erase(remove(out_dims.begin(), out_dims.end(), kDelFlag),
                     out_dims.end());
    }
    out->Resize(DDim(out_dims));
  }

  auto out_data = out->mutable_data<float>();
  int in_n = x_dims[0];
  int in_c = x_dims[1];
  int in_h = x_dims[2];
  int in_w = x_dims[3];

  if (dim.size() == 0) {
    reduce_mean_all(x_data, out_data, in_n, in_c, in_h, in_w);
  } else if (dim.size() == 1) {
    switch (dim[0]) {
      case 0:
        reduce_mean_n(x_data, out_data, in_n, in_c, in_h, in_w);
        break;
      case 1:
        reduce_mean_c(x_data, out_data, in_n, in_c, in_h, in_w);
        break;
      case 2:
        reduce_mean_h(x_data, out_data, in_n, in_c, in_h, in_w);
        break;
      case 3:
        reduce_mean_w(x_data, out_data, in_n, in_c, in_h, in_w);
        break;
      default:
        LOG(FATAL) << "error!!!";
    }
  } else if (dim.size() == 2) {
    if (dim[0] == 0 && dim[1] == 1) {
      reduce_mean_nc(x_data, out_data, in_n, in_c, in_h, in_w);
    } else if (dim[0] == 1 && dim[1] == 2) {
      reduce_mean_ch(x_data, out_data, in_n, in_c, in_h, in_w);
    } else if (dim[0] == 2 && dim[1] == 3) {
      reduce_mean_hw(x_data, out_data, in_n, in_c, in_h, in_w);
    } else {
      LOG(FATAL) << "invalid dim!!";
    }
  }
}

void test_reduce_mean(const std::vector<int64_t>& input_shape,
                      std::vector<int> dim,
                      bool keep_dim) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name("x");
  std::string out_var_name("out");
  std::string out_ref_var_name("out_ref");
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize(input_shape);

  // initialize input&output data
  FillTensor<float>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("reduce_mean");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("dim", dim);
  opdesc.SetAttr("keep_dim", keep_dim);

  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::ReduceMeanOp>(opdesc, &scope);
  LauchOp(op, {x_var_name}, {out_var_name});

  // execute reference implementation and save to output tensor
  reduce_mean_ref(op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(NPUBridges, reduce_mean) {
  std::vector<std::vector<int>> reduce_dim{
      {0}, {1}, {2}, {3}, {0, 1}, {1, 2}, {2, 3}, {-2, -1}};
  for (auto dim : reduce_dim) {
    for (auto keep_dim : {true, false}) {
      test_reduce_mean({1, 2, 3, 4}, dim, keep_dim);
    }
  }
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(reduce_mean);
USE_NPU_BRIDGE(reduce_mean);
