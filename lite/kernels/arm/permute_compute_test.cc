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

#include "lite/kernels/arm/permute_compute.h"
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void permute_compute_ref(const std::vector<lite::Tensor*> input,
                         std::vector<lite::Tensor*> output,
                         operators::PermuteParam param) {
  // const dtype* src_ptr = static_cast<const dtype*>(input[0] -> data());
  // dtype* dst_ptr = static_cast<dtype*>(output[0] -> mutable_data());
  const float* src_ptr = input[0]->data<float>();
  float* dst_ptr = output[0]->mutable_data<float>();

  std::vector<int> orders = param.order;
  // int out_size = output[0] ->size();
  int out_size = output[0]->dims().count(0, output[0]->dims().size());
  int num_axes = input[0]->dims().size();

  //    std::vector<int> new_steps = output[0] -> get_stride();
  //    std::vector<int> old_steps = input[0] -> get_stride();
  std::vector<int> old_steps;
  DDimLite input_dims = input[0]->dims();
  old_steps.resize(input_dims.size());
  for (int i = 0; i < input_dims.size(); ++i) {
    old_steps[i] = input_dims.count(i + 1, input_dims.size());
  }
  std::vector<int> new_steps;
  DDimLite output_dims = output[0]->dims();
  new_steps.resize(output_dims.size());
  for (int i = 0; i < output_dims.size(); ++i) {
    new_steps[i] = output_dims.count(i + 1, output_dims.size());
  }
  DDimLite new_valid_shape = output[0]->dims();

  for (int j = 0; j < out_size; ++j) {
    int in_idx = 0;
    int out_idx = 0;
    int new_valid_stride = 1;
    for (int i = num_axes - 1; i >= 0; --i) {
      int order = orders[i];
      int new_step = new_steps[i];
      int old_step = old_steps[order];
      int id = (j / new_valid_stride) % new_valid_shape[i];
      in_idx += id * old_step;
      out_idx += id * new_step;
      new_valid_stride *= new_valid_shape[i];
    }
    dst_ptr[out_idx] = src_ptr[in_idx];
  }
}

TEST(Permute_arm, init) {
  PermuteCompute permute;
  ASSERT_EQ(permute.precision(), PRECISION(kFloat));
  ASSERT_EQ(permute.target(), TARGET(kARM));
}

TEST(permute_arm, compute) {
  // 1、原始变量/////////
  PermuteCompute permute;
  operators::PermuteParam param;
  lite::Tensor tensorA;
  std::vector<lite::Tensor*> output;
  std::vector<lite::Tensor*> output_ref;
  output.push_back(new lite::Tensor);
  output_ref.push_back(new lite::Tensor);

  // 1、遍历index
  for (int s0 : {0, 1, 2, 3}) {
    for (int s1 : {0, 1, 2, 3}) {
      for (int s2 : {0, 1, 2, 3}) {
        for (int s3 : {0, 1, 2, 3}) {
          if (s0 != s1 && s0 != s2 && s0 != s3 && s1 != s2 && s1 != s3 &&
              s2 != s3) {
            LOG(INFO) << "(" << s0 << "," << s1 << "," << s2 << "," << s3
                      << ")";
            // PermuteParam<TargetType_D> param({s0, s1, s2, s3});
            param.order = {s0, s1, s2, s3};
            std::vector<int> v_n = {1, 2};
            std::vector<int> v_c = {1, 3};
            std::vector<int> v_h = {32, 64};
            std::vector<int> v_w = {32, 64};
            // mlu permute so so slow for now
            /* if (std::is_same<float, MLU>::value) {
                 v_n = {2};  v_c = {3};
                 v_h = {32}; v_w = {64};
             }*/
            for (int n : v_n) {
              for (int c : v_c) {
                for (int h : v_h) {
                  for (int w : v_w) {
                    // testbase.set_param(param);
                    // testbase.set_input_shape(Shape({n, c, h, w}));
                    // 2、遍历输入类型，给input赋值
                    for (auto out : output) delete out;
                    for (auto out : output_ref) delete out;
                    output.clear();
                    output_ref.clear();
                    LOG(INFO) << "1:......";
                    int outs_number = 1;
                    LOG(INFO) << "2:......";
                    // input data
                    DDimLite ddimA({n, c, h, w});
                    tensorA.Resize(ddimA);
                    for (int i = 0; i < ddimA.production(); i++) {
                      tensorA.mutable_data<float>()[i] = i;
                    }
                    LOG(INFO) << "3:......";
                    param.X.clear();
                    param.X.push_back(&tensorA);
                    //// output shape
                    std::vector<lite::DDim> input_dims;
                    for (auto p : param.X) {
                      input_dims.push_back(p->dims());
                    }
                    const size_t n = input_dims.size();
                    for (int i = 0; i < n; i++) {
                      auto& out_dims = input_dims[i];
                      CHECK_EQ(input_dims[i].size(), param.order.size())
                          << "permute order param is not valid";
                      for (int j = 0; j < param.order.size(); j++) {
                        out_dims[j] = input_dims[i][param.order[j]];
                      }
                      LOG(INFO) << "3:......";
                      output.push_back(new lite::Tensor);
                      LOG(INFO) << "4:......";
                      output[i]->Resize(lite::DDim(out_dims));
                      LOG(INFO) << "5:......";
                      output_ref.push_back(new lite::Tensor);
                      LOG(INFO) << "6:......";
                      output_ref[i]->Resize(lite::DDim(out_dims));
                      auto* output_data = output[i]->mutable_data<float>();
                      auto* output_ref_data =
                          output_ref[i]->mutable_data<float>();
                      for (int k = 0; k < out_dims.production(); ++k) {
                        output_data[k] = -2;
                        output_ref_data[k] = -2;
                      }
                      // param_.Out[i]->Resize(lite::DDim(out_dims));
                    }

                    /*                                          output.Resize(ddimA);
                                                                output_ref.Resize(ddimA);
                                                                auto*
                       output_data = output.mutable_data<float>();
                                                                auto*
                       output_ref_data = output_ref.mutable_data<float>();
                                                                for (int i = 0;
                       i < ddimA.production(); ++i) {
                                                                    output_data[i]
                       = -2;
                                                                    output_ref_data[i]
                       = -2;
                                                                }*/
                    param.Out = output;
                    permute.SetParam(param);
                    // param.X[0] = &tensorA;
                    // param.X[0] = &tensorA;//有问题，应该用append
                    LOG(INFO) << "7:......";
                    // 3 set the input
                    std::unique_ptr<KernelContext> ctx(new KernelContext);
                    ctx->As<ARMContext>();
                    DeviceInfo::Init();
                    permute.SetContext(std::move(ctx));
                    permute.PrepareForRun();
                    LOG(INFO) << "8:....";
                    permute.Run();
                    LOG(INFO) << "permute.Run end";
                    param.Out = output_ref;
                    LOG(INFO) << "permute_compute_ref start";
                    permute_compute_ref(param.X, output_ref, param);
                    LOG(INFO) << "permute_compute_ref end";
                    // 4、检查结果是否一致
                    for (int i = 0; i < output.size(); i++) {
                      float* output_data = output[i]->mutable_data<float>();
                      float* output_ref_data =
                          output_ref[i]->mutable_data<float>();
                      for (int j = 0; j < output[i]->dims().production(); j++) {
                        EXPECT_NEAR(output_data[j], output_ref_data[j], 1e-5);
                      }
                    }
                  }
                }
              }
            }  // end for nchw
          }
        }
      }
    }
  }  // end for permute
}
TEST(permute, retrive_op) {
  auto permute =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "permute");
  ASSERT_FALSE(permute.empty());
  ASSERT_TRUE(permute.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(permute, kARM, kFloat, kNCHW, def);
