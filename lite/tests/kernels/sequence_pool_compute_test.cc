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
#include "lite/core/test/arena/framework.h"

namespace paddle {
namespace lite {

class SequencePoolComputeTester : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string input_ = "x";
  std::string output_ = "out";
  LoD lod_{{0, 2, 5}};
  std::string pool_type_ = "SUM";
  DDim dims_{{5, 1}};

 public:
  SequencePoolComputeTester(const Place& place,
                            const std::string& alias,
                            LoD lod,
                            std::string pool_type,
                            DDim dims)
      : TestCase(place, alias), lod_(lod), pool_type_(pool_type), dims_(dims) {}

  void RunBaseline(Scope* scope) override {
    auto* x = scope->FindMutableTensor(input_);
    const auto* x_data = x->data<float>();
    (x->mutable_lod())->clear();
    (x->mutable_lod())->push_back(lod_[0]);
    auto seq_offset = x->lod()[0];
    int width = x->numel() / dims_[0];
    std::vector<int64_t> out_dims;
    for (int i = 0; i < dims_.size(); i++) {
      out_dims.push_back(dims_[i]);
    }
    out_dims[0] = x->lod()[0].size() - 1;
    auto* out = scope->NewTensor(output_);
    out->Resize(out_dims);
    auto* out_data = out->mutable_data<float>();

    for (int i = 0; i < seq_offset.size() - 1; i++) {
      int slice_num = seq_offset[i + 1] - seq_offset[i];
      const float* x_data_ptr = x_data + seq_offset[i] * width;
      float* out_data_ptr = out_data + i * width;
      if (slice_num > 0) {
        if (pool_type_ == "SUM") {
          for (int j = 0; j < width; ++j) {
            float sum = x_data_ptr[j];
            for (int k = 1; k < slice_num; ++k) {
              float x_data_read = x_data_ptr[k * width + j];
              sum += x_data_read;
            }
            out_data_ptr[j] = sum;
          }
        } else if (pool_type_ == "AVERAGE") {
          for (int j = 0; j < width; ++j) {
            float sum = x_data_ptr[j];
            for (int k = 1; k < slice_num; ++k) {
              float x_data_read = x_data_ptr[k * width + j];
              sum += x_data_read;
            }
            out_data_ptr[j] = sum / slice_num;
          }
        } else if (pool_type_ == "SQRT") {
          float sqrt_len = sqrtf(slice_num);
          for (int j = 0; j < width; ++j) {
            float sum = x_data_ptr[j];
            for (int k = 1; k < slice_num; ++k) {
              float x_data_read = x_data_ptr[k * width + j];
              sum += x_data_read;
            }
            out_data_ptr[j] = sum / sqrt_len;
          }
        } else if (pool_type_ == "MAX") {
          for (int j = 0; j < width; ++j) {
            float max = x_data_ptr[j];
            for (int k = 1; k < slice_num; ++k) {
              float x_data_read = x_data_ptr[k * width + j];
              if (max < x_data_read) {
                max = x_data_read;
              }
            }
            out_data_ptr[j] = max;
          }
        } else if (pool_type_ == "MIN") {
          for (int j = 0; j < width; ++j) {
            float min = x_data_ptr[j];
            for (int k = 1; k < slice_num; ++k) {
              float x_data_read = x_data_ptr[k * width + j];
              if (min > x_data_read) {
                min = x_data_read;
              }
            }
            out_data_ptr[j] = min;
          }
        } else if (pool_type_ == "FIRST") {
          memcpy(out_data_ptr, x_data_ptr, width * sizeof(float));
        } else if (pool_type_ == "LAST") {
          int64_t seq_len =
              static_cast<int64_t>(seq_offset[i + 1] - seq_offset[0]);
          x_data_ptr = x_data + width * seq_len;
          memcpy(out_data_ptr, x_data_ptr - width, width * sizeof(float));
        } else {
          LOG(ERROR) << " UNKNOWN seq pool type";
        }
      }
    }
    int batch_size = seq_offset.size() - 1;
    std::vector<uint64_t> offset_new(static_cast<uint64_t>(batch_size + 1));
    for (int i = 0; i <= batch_size; i++) {
      offset_new[i] = i;
    }
    (out->mutable_lod())->push_back(offset_new);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("sequence_pool");
    op_desc->SetInput("X", {input_});
    op_desc->SetOutput("Out", {output_});
    op_desc->SetAttr("pooltype", pool_type_);
  }

  void PrepareData() override {
    std::vector<float> data(dims_.production());

    for (int i = 0; i < dims_.production(); i++) {
      data[i] = i * 1.1;
    }
    SetCommonTensor(input_, dims_, data.data(), lod_);
  }
};
void generate_lod(int seq_num,
                  int max_len,
                  std::vector<uint64_t>& seq_offset) {  // NOLINT
  seq_offset.clear();
  int sum = 0;
  seq_offset.push_back(sum);
  for (int i = 0; i < seq_num; i++) {
    sum += std::rand() % max_len + 1;
    seq_offset.push_back(uint64_t(sum));
  }
}

void test_sequence_pool(Place place) {
  int max_len = 2;
  for (auto c : {1, 3, 4}) {
    for (auto h : {1, 3, 4}) {
      for (auto w : {1, 3, 4}) {
        for (auto pool_type :
#if defined(LITE_WITH_XPU)
             {"SUM", "MAX", "FIRST", "LAST"}) {
#else
             {"SUM", "AVERAGE", "SQRT", "MAX", "MIN", "FIRST", "LAST"}) {
#endif
          for (int seq_num : {1, 3, 5}) {
            std::vector<std::vector<uint64_t>> lod;
            lod.resize(1);
            generate_lod(seq_num, max_len, lod[0]);
            int64_t n = int64_t(lod[0].back());
            auto dims = DDim(std::vector<int64_t>({n, c, h, w}));
            std::unique_ptr<arena::TestCase> tester(
                new SequencePoolComputeTester(
                    place, "def", lod, pool_type, dims));
            arena::Arena arena(std::move(tester), place, 2e-5);
            arena.TestPrecision();
          }
        }
      }
    }
  }
}

TEST(SequencePool, precision) {
// #ifdef LITE_WITH_X86
//   Place place(TARGET(kX86));
// #endif
#if defined(LITE_WITH_XPU)
  Place place(TARGET(kXPU));
#elif defined(LITE_WITH_ARM)
  Place place(TARGET(kARM));
#endif
  test_sequence_pool(place);
}

}  // namespace lite
}  // namespace paddle
