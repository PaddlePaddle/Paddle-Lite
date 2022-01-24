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
#include <stdlib.h>
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/core/test/arena/framework.h"

namespace paddle {
namespace lite {

void reduce_all_n(const bool* src,
                  bool* dst,
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
        dst[data_index] = true;
        for (int n = 0; n < num_in; ++n) {
          src_index = n * chw_size + data_index;
          dst[data_index] = dst[data_index] && src[src_index];
        }
      }
    }
  }
}

void reduce_all_c(const bool* src,
                  bool* dst,
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
        dst[data_index] = true;
        for (int c = 0; c < channel_in; ++c) {
          src_index = src_index0 + c * hw_size;
          dst[data_index] = dst[data_index] && src[src_index];
        }
      }
    }
  }
}

void reduce_all_h(const bool* src,
                  bool* dst,
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
        dst[data_index] = true;
        for (int h = 0; h < height_in; ++h) {
          src_index = src_index0 + h * width_in;
          dst[data_index] = dst[data_index] && src[src_index];
        }
      }
    }
  }
}

void reduce_all_w(const bool* src,
                  bool* dst,
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
        dst[data_index] = true;
        for (int w = 0; w < width_in; ++w) {
          src_index = src_index0 + w;
          dst[data_index] = dst[data_index] && src[src_index];
        }
      }
    }
  }
}

void reduce_all_all(const bool* src,
                    bool* dst,
                    int num_in,
                    int channel_in,
                    int height_in,
                    int width_in) {
  bool all = true;
  int src_index;
  int n_id, c_id;
  for (int n = 0; n < num_in; ++n) {
    n_id = n * channel_in * height_in * width_in;
    for (int c = 0; c < channel_in; ++c) {
      c_id = c * height_in * width_in;
      for (int h = 0; h < height_in; ++h) {
        for (int w = 0; w < width_in; ++w) {
          src_index = n_id + c_id + h * width_in + w;
          all = all && src[src_index];
        }
      }
    }
  }
  dst[0] = all;
}

void reduce_all_nc(const bool* src,
                   bool* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  // reduce n first.
  DDimLite ddimA({1, channel_in, height_in, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  bool* tmp_out = tensor_tmp.mutable_data<bool>();
  reduce_all_n(src, tmp_out, num_in, channel_in, height_in, width_in);
  reduce_all_c(tmp_out, dst, 1, channel_in, height_in, width_in);
}

void reduce_all_ch(const bool* src,
                   bool* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  // reduce c first
  DDimLite ddimA({num_in, 1, height_in, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  bool* tmp_out = tensor_tmp.mutable_data<bool>();
  reduce_all_c(src, tmp_out, num_in, channel_in, height_in, width_in);
  reduce_all_h(tmp_out, dst, num_in, 1, height_in, width_in);
}

void reduce_all_hw(const bool* src,
                   bool* dst,
                   int num_in,
                   int channel_in,
                   int height_in,
                   int width_in) {
  // reduce h first
  DDimLite ddimA({num_in, channel_in, 1, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  bool* tmp_out = tensor_tmp.mutable_data<bool>();
  reduce_all_h(src, tmp_out, num_in, channel_in, height_in, width_in);
  reduce_all_w(tmp_out, dst, num_in, channel_in, 1, width_in);
}

class ReduceAllComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  std::vector<int> dim_{0};
  bool keep_dim_ = false;
  DDim x_dims_{{3, 2, 3, 4}};
  bool reduce_all_ = false;

 public:
  ReduceAllComputeTester(const Place& place,
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
    const auto* x_data = x->data<bool>();
    auto* out = scope->NewTensor(output_);
    size_t x_rank = x_dims_.size();
    if (!dim_.empty()) {
      for (size_t i = 0; i < dim_.size(); i++) {
        if (dim_[i] < 0) {
          dim_[i] += x_rank;
        }
      }
    }

    std::set<int> dims_set(dim_.begin(), dim_.end());
    bool full_dim = true;
    for (size_t i = 0; i < x_rank; i++) {
      if (dims_set.find(i) == dims_set.end()) {
        full_dim = false;
        break;
      }
    }
    reduce_all_ = (reduce_all_ || full_dim);
    if (dim_.size() == 0) {
      reduce_all_ = true;
    }

    std::vector<int64_t> out_dims;
    if (reduce_all_) {
      if (keep_dim_)
        out_dims = std::vector<int64_t>(x_rank, 1);
      else
        out_dims = std::vector<int64_t>{1};
    } else {
      size_t out_rank = keep_dim_ ? x_rank : x_rank - dim_.size();
      out_dims.resize(out_rank);
      std::stable_sort(dim_.begin(), dim_.end());
      int dim_index = 0;
      int out_index = 0;
      for (size_t i = 0; i < x_rank; ++i) {
        if (dim_index < static_cast<int>(dim_.size()) &&
            dim_[dim_index] == static_cast<DDim::value_type>(i)) {
          if (keep_dim_) {
            out_dims[out_index++] = 1;
          }
          dim_index++;
        } else {
          out_dims[out_index++] = x_dims_[i];
        }
      }
    }
    out->Resize(DDim(out_dims));

    auto* out_data = out->mutable_data<bool>();
    size_t new_dims[] = {1, 1, 1, 1};
    for (size_t j = 0; j < x_dims_.size(); ++j) {
      new_dims[j] = x_dims_[j];
    }
    int in_n = new_dims[0];
    int in_c = new_dims[1];
    int in_h = new_dims[2];
    int in_w = new_dims[3];

    if (reduce_all_) {
      reduce_all_all(x_data, out_data, in_n, in_c, in_h, in_w);
    } else if (dim_.size() == 1) {
      switch (dim_[0]) {
        case 0:
          reduce_all_n(x_data, out_data, in_n, in_c, in_h, in_w);
          break;
        case 1:
          reduce_all_c(x_data, out_data, in_n, in_c, in_h, in_w);
          break;
        case 2:
          reduce_all_h(x_data, out_data, in_n, in_c, in_h, in_w);
          break;
        case 3:
          reduce_all_w(x_data, out_data, in_n, in_c, in_h, in_w);
          break;
        default:
          LOG(FATAL) << "error!!!";
      }
    } else if (dim_.size() == 2) {
      if (dim_[0] == 0 && dim_[1] == 1) {
        reduce_all_nc(x_data, out_data, in_n, in_c, in_h, in_w);
      } else if (dim_[0] == 1 && dim_[1] == 2) {
        reduce_all_ch(x_data, out_data, in_n, in_c, in_h, in_w);
      } else if (dim_[0] == 2 && dim_[1] == 3) {
        reduce_all_hw(x_data, out_data, in_n, in_c, in_h, in_w);
      } else {
        LOG(FATAL) << "invalid dims_!!";
      }
    }
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("reduce_all");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("dim", dim_);
    op_desc->SetAttr("keep_dim", keep_dim_);
  }

  void PrepareData() override {
    std::vector<uint8_t> data(x_dims_.production());
    for (int i = 0; i < x_dims_.production(); i++) {
      data[i] = 1;  // static_cast<float>(rand()%2);
    }
    SetCommonTensor(input_, x_dims_, data.data());
  }
};

void test_reduce_all(Place place, float abs_err) {
  std::vector<std::vector<int>> reduce_dim{
      {0}, {1}, {2}, {3}, {0, 1}, {1, 2}, {2, 3}, {-2, -1}};
  for (auto n : {1, 3}) {
    for (auto c : {1, 2}) {
      for (auto h : {1, 3}) {
        for (auto w : {1, 3}) {
          for (bool keep_dim : {false, true}) {
            for (auto dim : reduce_dim) {
              DDim x_dims;
              for (auto dims : {2, 3, 4}) {
                switch (dims) {
                  case 2:
                    x_dims = DDim(std::vector<int64_t>({n, c}));
                    break;
                  case 3:
                    x_dims = DDim(std::vector<int64_t>({n, c, h}));
                    break;
                  case 4:
                    x_dims = DDim(std::vector<int64_t>({n, c, h, w}));
                    break;
                  default:
                    x_dims = DDim(std::vector<int64_t>({n, c, h, w}));
                }

                int last_dim = dim.back();
                if (dim.back() < 0) {
                  last_dim += x_dims.size();
                  if (last_dim < 1) continue;
                }
                if (last_dim > x_dims.size() - 1) continue;

#ifdef LITE_WITH_OPENCL
                // fixme: currently utest will fail when keep_dim == false on
                // same case(such as nchw{1,2,1,1}, dim{2}). Not that the kernel
                // is right on this case but the utest will fail because cannot
                // get the padded dims of output tensor in framework.cc
                keep_dim = true;
#endif
                std::unique_ptr<arena::TestCase> tester(
                    new ReduceAllComputeTester(
                        place, "def", dim, keep_dim, x_dims));
                arena::Arena arena(std::move(tester), place, abs_err);
                arena.TestPrecision();
              }
            }
          }
        }
      }
    }
  }
}

TEST(ReduceAll, precision) {
#if defined(LITE_WITH_XPU)
  Place place(TARGET(kXPU));
  float abs_err = 2e-5;
  test_reduce_all(place, abs_err);
#elif defined(LITE_WITH_ARM) || defined(LITE_WITH_X86)
  Place place(TARGET(kHost));
  float abs_err = 2e-5;
  test_reduce_all(place, abs_err);
#endif
}

}  // namespace lite
}  // namespace paddle
