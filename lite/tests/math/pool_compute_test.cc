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

#include "lite/tests/math/pool_ut.h"

#ifdef LITE_WITH_ARM
void test_pool_fp32(const std::vector<DDim>& input_dims,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    bool ceil_mode,
                    bool flag_global,
                    bool exclusive,
                    bool adaptive,
                    bool use_quantizer,
                    std::string pooling_type,
                    const std::vector<int>& thread_num,
                    const std::vector<int>& power_mode) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  PoolParam param;
  param.x = new Tensor;
  param.x->set_precision(PRECISION(kFloat));
  param.ksize = ksize;

  param.strides = strides;
  param.paddings = std::make_shared<std::vector<int>>(pads);
  param.ceil_mode = ceil_mode;
  param.global_pooling = flag_global;
  param.pooling_type = pooling_type;
  param.exclusive = exclusive;
  param.adaptive = adaptive;
  param.use_quantizer = use_quantizer;

  param.output = new Tensor;
  param.output->set_precision(PRECISION(kFloat));

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::PoolCompute<PRECISION(kFloat),
                                              PRECISION(kFloat)>
          pool;
      std::unique_ptr<paddle::lite::KernelContext> ctx1(
          new paddle::lite::KernelContext);
      auto& ctx = ctx1->As<paddle::lite::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_api::PowerMode>(cls), th);
      /// set param and context
      pool.SetParam(param);
      pool.SetContext(std::move(ctx1));
      /// prepare for run
      pool.PrepareForRun();

      for (auto& dim_in : input_dims) {
        DDim dim_out = compute_out_dim(dim_in, param);
        if (dim_out[2] < 1 || dim_out[3] < 1) {
          continue;
        }
        param.x->Resize(dim_in);
        param.output->Resize(dim_out);

        paddle::lite::fill_tensor_rand(*param.x, -1.f, 1.f);
        //        paddle::lite::fill_tensor_const(*param.x, 1.f);
        auto din = param.x->data<float>();

        Tensor tout_basic;
        if (FLAGS_check_result) {
          LOG(INFO) << "basic compute";
          tout_basic.set_precision(PRECISION(kFloat));
          tout_basic.Resize(dim_out);
          fill_tensor_const(tout_basic, 0.f);
          auto dout_basic = tout_basic.mutable_data<float>();
          pooling_basic<float, float>(din,
                                      dout_basic,
                                      dim_in[0],
                                      dim_out[1],
                                      dim_out[2],
                                      dim_out[3],
                                      dim_in[1],
                                      dim_in[2],
                                      dim_in[3],
                                      ksize,
                                      strides,
                                      pads,
                                      flag_global,
                                      exclusive,
                                      adaptive,
                                      ceil_mode,
                                      use_quantizer,
                                      pooling_type);
        }
        LOG(INFO) << "lite compute";
        /// warm up
        for (int i = 0; i < FLAGS_warmup; ++i) {
          pool.Launch();
        }
        /// compute
        Timer t0;
        for (int i = 0; i < FLAGS_repeats; ++i) {
          t0.Start();
          pool.Launch();
          t0.Stop();
        }

        double gops = 2.0 * dim_out.production() * ksize[0] * ksize[1];

        print_gops_info("pool_fp32", dim_in, dim_out, t0, gops);
        if (FLAGS_check_result) {
          double max_ratio = 0;
          double max_diff = 0;
          tensor_cmp_host(tout_basic, *param.output, max_ratio, max_diff);
          print_diff_info(max_diff, max_ratio);
          if (std::abs(max_ratio) > 1e-3f) {
            if (max_diff > 5e-4f) {
              print_tensor_info_common(
                  *param.x, tout_basic, *param.output, true);
              print_pool_success_or_fail_info("pool_fp32",
                                              false,
                                              dim_in,
                                              dim_out,
                                              ksize,
                                              pads,
                                              strides,
                                              flag_global,
                                              pooling_type,
                                              ceil_mode,
                                              exclusive,
                                              th,
                                              cls);
            }
          }
        }
        print_pool_success_or_fail_info("pool_fp32",
                                        true,
                                        dim_in,
                                        dim_out,
                                        ksize,
                                        pads,
                                        strides,
                                        flag_global,
                                        pooling_type,
                                        ceil_mode,
                                        exclusive,
                                        th,
                                        cls);
      }
    }
  }

  delete param.x;
  delete param.output;
}
#else
void test_pool_fp32(const std::vector<DDim>& input_dims,
                    const std::vector<int>& ksize,
                    const std::vector<int>& strides,
                    const std::vector<int>& pads,
                    bool ceil_mode,
                    bool flag_global,
                    bool exclusive,
                    bool adaptive,
                    bool use_quantizer,
                    std::string pooling_type,
                    const std::vector<int>& thread_num,
                    const std::vector<int>& power_mode) {}
#endif  // LITE_WITH_ARM

#if 1  /// random param pool
TEST(TestPoolRand, test_pool_rand) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8, 16}) {
      for (auto& kw : {1, 2, 3}) {
        for (auto& kh : {1, 2, 3}) {
          for (auto& stride : {1, 2}) {
            for (auto& pad_top : {0, 1, 2}) {
              for (auto& pad_bottom : {0, 1, 2}) {
                for (auto& pad_left : {0, 1, 2}) {
                  for (auto& pad_right : {0, 1, 2}) {
                    for (auto& flag_global : {false, true}) {
                      for (auto& exclusive : {false, true}) {
                        for (auto& ceil_mode : {false, true}) {
                          for (auto& pooling_type : {"max", "avg"}) {
                            bool adaptive = false;
                            bool use_quantizer = false;
                            std::vector<DDim> dims;
                            for (auto& batch : {1, 2}) {
                              for (auto& h : {1, 2, 3, 4, 11, 19, 32, 28}) {
                                dims.push_back(DDim({batch, cin, h, h}));
                              }
                            }
                            test_pool_fp32(
                                dims,
                                {kh, kw},
                                {stride, stride},
                                {pad_top, pad_bottom, pad_left, pad_right},
                                ceil_mode,
                                flag_global,
                                exclusive,
                                adaptive,
                                use_quantizer,
                                pooling_type,
                                {4},
                                {FLAGS_power_mode});
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
  }
}
#endif  /// random param conv

#ifdef LITE_WITH_ARM8_SVE2  // global_pool
TEST(TesPoolGlobal, test_pool_fp32_global) {
  for (auto& h : {51})
    test_pool_fp32({DDim({1, 64, h, h})},
                   {2, 2},
                   {1, 1},
                   {1, 1, 1, 1},
                   false,
                   true,
                   false,
                   false,
                   false,
                   "avg",
                   {1},
                   {1});
}
#endif  // global_pool

#if 1  /// custom
TEST(TesPoolCustom, test_pool_fp32_custom_size) {
  test_pool_fp32(
      {DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width})},
      {FLAGS_kernel_h, FLAGS_kernel_w},
      {FLAGS_stride_h, FLAGS_stride_w},
      {FLAGS_pad_h, FLAGS_pad_h, FLAGS_pad_w, FLAGS_pad_w},
      FLAGS_ceil_mode,
      FLAGS_flag_global,
      FLAGS_exclusive,
      FLAGS_adaptive,
      FLAGS_use_quantizer,
      FLAGS_pooling_type,
      {FLAGS_threads},
      {FLAGS_power_mode});
}
#endif  // custom
