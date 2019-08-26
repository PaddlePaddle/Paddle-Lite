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
#include "lite/tests/kernels/fill_data.h"
#include "lite/tests/kernels/test_funcs.h"

namespace paddle {
namespace lite {

int pool_output_size(
    int input_size, int filter_size, int padding, int stride, bool ceil_mode) {
  int output_size;
  if (!ceil_mode) {
    output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  } else {
    output_size =
        (input_size - filter_size + 2 * padding + stride - 1) / stride + 1;
  }
  return output_size;
}

DDim compute_pool_shape(const DDim& x_dims,
                        const std::vector<int>& ksize,
                        const std::vector<int>& strides,
                        const std::vector<int>& pads,
                        bool global,
                        bool ceil_mode) {
  std::vector<int64_t> out_shape;
  for (int j = 0; j < x_dims.size() - ksize.size(); ++j) {
    out_shape.push_back(x_dims[j]);
  }
  if (global) {
    for (int j = 0; j < ksize.size(); ++j) {
      out_shape.push_back(1);
    }
    return DDim(out_shape);
  }

  for (size_t i = 0; i < ksize.size(); ++i) {
    out_shape.push_back(pool_output_size(
        x_dims[i + 2], ksize[i], pads[i], strides[i], ceil_mode));
  }
  return DDim(out_shape);
}

class PoolOPTest : public arena::TestCase {
 protected:
  // common attributes for this op.
  std::string x_ = "X";
  std::string out_ = "Out";
  std::string pooling_type_ = "max";
  std::vector<int> ksize_ = {2, 2};
  std::vector<int> strides_ = {2, 2};
  std::vector<int> paddings_ = {0, 0};
  bool global_pooling_ = false;
  bool exclusive_ = false;
  bool ceil_mode_ = false;
  DDim dims_{{1, 32, 112, 112}};

 public:
  PoolOPTest(const Place& place,
             const std::string& alias,
             const DDim& dim_in,
             const std::string& pool_type,
             const std::vector<int>& ksize,
             const std::vector<int>& strides,
             const std::vector<int>& pads,
             bool global,
             bool exclusive,
             bool ceil_mode)
      : TestCase(place, alias),
        dims_(std::move(dim_in)),
        pooling_type_(pool_type),
        ksize_(ksize),
        strides_(strides),
        paddings_(pads),
        global_pooling_(global),
        exclusive_(exclusive),
        ceil_mode_(ceil_mode) {}

  void RunBaseline(Scope* scope) override {
    auto x = scope->FindTensor(x_);
    auto out = scope->NewTensor(out_);
    CHECK(out);
    LOG(INFO) << "input dims: " << x->dims();
    DDim out_dim = compute_pool_shape(
        x->dims(), ksize_, strides_, paddings_, global_pooling_, ceil_mode_);
    out->Resize(out_dim);

    LOG(INFO) << "out dims: " << out_dim;

    auto x_data = x->data<float>();
    auto out_data = out->mutable_data<float>();

    pooling_basic(x_data,
                  out_data,
                  x->dims()[0],
                  x->dims()[1],
                  out->dims()[2],
                  out->dims()[3],
                  x->dims()[1],
                  x->dims()[2],
                  x->dims()[3],
                  ksize_,
                  strides_,
                  paddings_,
                  global_pooling_,
                  exclusive_,
                  false,
                  ceil_mode_,
                  false,
                  pooling_type_);
  }

  void PrepareOpDesc(cpp::OpDesc* op_desc) {
    op_desc->SetType("pool2d");
    op_desc->SetInput("X", {x_});
    op_desc->SetOutput("Out", {out_});
    op_desc->SetAttr<std::vector<int>>("ksize", ksize_);
    op_desc->SetAttr<std::vector<int>>("strides", strides_);
    op_desc->SetAttr<std::vector<int>>("paddings", paddings_);
    op_desc->SetAttr<std::string>("pooling_type", pooling_type_);
    op_desc->SetAttr<bool>("global_pooling", global_pooling_);
    op_desc->SetAttr<bool>("exclusive", exclusive_);
    op_desc->SetAttr<bool>("ceil_mode", ceil_mode_);
  }

  void PrepareData() override {
    std::vector<float> din(dims_.production());
    fill_data_rand(din.data(), -1.f, 1.f, dims_.production());
    SetCommonTensor(x_, dims_, din.data());
  }
};

void test_pool(Place place) {
  for (auto n : {1, 2}) {
    for (auto c : {1, 3, 32, 48}) {
      for (auto h : {2, 3, 4, 11, 17, 28, 32, 75}) {
        for (auto ksize : {2, 3}) {
          for (auto stride : {1, 2}) {
            for (auto pad : {0, 1}) {
              for (auto pooling_type : {"max", "avg"}) {
                for (auto ceil_mode : {true, false}) {
                  for (auto global_pooling : {true, false}) {
                    for (auto exclusive : {true, false}) {
                      for (auto th : {1, 2, 4}) {
                        int w = h;
                        DDim dim_in(std::vector<int64_t>{n, c, h, w});
                        auto dim_out =
                            compute_pool_shape(dim_in,
                                               std::vector<int>{ksize, ksize},
                                               std::vector<int>{stride, stride},
                                               std::vector<int>{pad, pad},
                                               global_pooling,
                                               ceil_mode);
                        if (dim_out.production() <= 0) {
                          continue;
                        }
                        std::unique_ptr<arena::TestCase> tester(
                            new PoolOPTest(place,
                                           "def",
                                           dim_in,
                                           pooling_type,
                                           std::vector<int>{ksize, ksize},
                                           std::vector<int>{stride, stride},
                                           std::vector<int>{pad, pad},
                                           global_pooling,
                                           exclusive,
                                           ceil_mode));
#ifdef LITE_WITH_ARM
                        auto& ctx = tester->context()->As<ARMContext>();
                        ctx.SetRunMode(lite_api::LITE_POWER_HIGH, th);
#endif
                        arena::Arena arena(std::move(tester), place, 6e-5);
                        if (!arena.TestPrecision()) {
                          LOG(ERROR) << "n:" << n << " c:" << c << " h:" << h
                                     << " w:" << w << " ksize:" << ksize
                                     << " stride:" << stride << " pad:" << pad
                                     << " exclusive:" << exclusive
                                     << " global_pooling:" << global_pooling
                                     << " ceil_mode: " << ceil_mode
                                     << " pooling_type:" << pooling_type
                                     << " threads: " << th << " failed";
                          return;
                        }
                        LOG(INFO) << "n:" << n << " c:" << c << " h:" << h
                                  << " w:" << w << " ksize:" << ksize
                                  << " stride:" << stride << " pad:" << pad
                                  << " exclusive:" << exclusive
                                  << " global_pooling:" << global_pooling
                                  << " ceil_mode: " << ceil_mode
                                  << " pooling_type:" << pooling_type
                                  << " threads: " << th << " successed";
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

TEST(PoolOPTest, precision) {
#ifdef LITE_WITH_X86
  Place place(TARGET(kX86));
#endif
#ifdef LITE_WITH_ARM
  Place place(TARGET(kARM));
  test_pool(place);
#endif
}

}  // namespace lite
}  // namespace paddle
