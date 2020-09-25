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

#include <gtest/gtest.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/arena/framework.h"

namespace paddle {
namespace lite {

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

class ReduceMeanComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  std::vector<int> dim_{0};
  bool keep_dim_ = false;
  DDim x_dims_{{3, 2, 3, 4}};
  bool reduce_all_ = false;

 public:
  ReduceMeanComputeTester(const Place& place,
                          const std::string& alias,
                          std::vector<int> dim,
                          bool keep_dim,
                          DDim x_dims)
      : TestCase(place, alias),
        dim_(dim),
        keep_dim_(keep_dim),
        x_dims_(x_dims) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindMutableTensor(input_);
    const auto* x_data = x->data<float>();
    auto* out = scope->NewTensor(output_);
    auto x_rank = x_dims_.size();
    if (!dim_.empty()) {
      for (int i = 0; i < dim_.size(); i++) {
        if (dim_[i] < 0) {
          dim_[i] += x_rank;
        }
      }
    }

    std::stable_sort(dim_.begin(), dim_.end());
    if (dim_.size() == 0) {
      reduce_all_ = true;
    }
    std::vector<int64_t> out_dims;
    if (reduce_all_) {
      if (keep_dim_) {
        out_dims.push_back(x_rank);
        out_dims.push_back(1);
      } else {
        out_dims.push_back(1);
      }
    } else {
      for (int i = 0; i < x_dims_.size(); i++) {
        out_dims.push_back(x_dims_[i]);
      }
      if (keep_dim_) {
        for (size_t i = 0; i < dim_.size(); ++i) {
          out_dims[dim_[i]] = 1L;
        }
      } else {
        int64_t kDelFlag = -2;
        for (size_t i = 0; i < dim_.size(); ++i) {
          out_dims[dim_[i]] = kDelFlag;
        }
        out_dims.erase(remove(out_dims.begin(), out_dims.end(), kDelFlag),
                       out_dims.end());
      }
      out->Resize(DDim(out_dims));
    }

    auto* out_data = out->mutable_data<float>();
    int in_n = x_dims_[0];
    int in_c = x_dims_[1];
    int in_h = x_dims_[2];
    int in_w = x_dims_[3];

    if (dim_.size() == 0) {
      reduce_mean_all(x_data, out_data, in_n, in_c, in_h, in_w);
    } else if (dim_.size() == 1) {
      switch (dim_[0]) {
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
    } else if (dim_.size() == 2) {
      if (dim_[0] == 0 && dim_[1] == 1) {
        reduce_mean_nc(x_data, out_data, in_n, in_c, in_h, in_w);
      } else if (dim_[0] == 1 && dim_[1] == 2) {
        reduce_mean_ch(x_data, out_data, in_n, in_c, in_h, in_w);
      } else if (dim_[0] == 2 && dim_[1] == 3) {
        reduce_mean_hw(x_data, out_data, in_n, in_c, in_h, in_w);
      } else {
        LOG(FATAL) << "invalid dims_!!";
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("reduce_mean");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("dim", dim_);
    op_desc->SetAttr("keep_dim", keep_dim_);
  }

  void PrepareData() override {
    std::vector<float> data(x_dims_.production());
    for (int i = 0; i < x_dims_.production(); i++) {
      data[i] = i * 1.0;
    }
    SetCommonTensor(input_, x_dims_, data.data());
  }
};

void test_reduce_mean(Place place) {
  std::vector<std::vector<int>> reduce_dim{
      {0}, {1}, {2}, {3}, {0, 1}, {1, 2}, {2, 3}, {-2, -1}};
  for (auto n : {1, 3}) {
    for (auto c : {1, 2}) {
      for (auto h : {1, 3}) {
        for (auto w : {1, 3}) {
          for (bool keep_dim : {false, true}) {
            for (auto dim : reduce_dim) {
              auto x_dims = DDim(std::vector<int64_t>({n, c, h, w}));
              std::unique_ptr<arena::TestCase> tester(
                  new ReduceMeanComputeTester(
                      place, "def", dim, keep_dim, x_dims));
              arena::Arena arena(std::move(tester), place, 2e-5);
              arena.TestPrecision();
            }
          }
        }
      }
    }
  }
}

TEST(ReduceMean, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
  test_reduce_mean(place);
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_reduce_mean(place);
#endif
}

}  // namespace lite
}  // namespace paddle
