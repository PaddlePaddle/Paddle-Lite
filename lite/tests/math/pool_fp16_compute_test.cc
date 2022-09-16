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
void test_pool_fp16(const std::vector<DDim>& input_dims,
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
                    const std::vector<int>& power_mode,
                    std::vector<DDim> output_dims = {}) {
#ifdef LITE_WITH_ARM
  paddle::lite::DeviceInfo::Init();
#endif
  PoolParam param;
  param.x = new Tensor;
  param.x->set_precision(PRECISION(kFP16));
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
  param.output->set_precision(PRECISION(kFP16));

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite::kernels::arm::PoolCompute<PRECISION(kFP16),
                                              PRECISION(kFP16)>
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
        if (adaptive) {
          dim_out = output_dims[0];
        }
        if (dim_out[2] < 1 || dim_out[3] < 1) {
          continue;
        }
        param.x->Resize(dim_in);
        param.output->Resize(dim_out);

        Tensor x_fp32;
        x_fp32.Resize(dim_in);
        x_fp32.set_precision(PRECISION(kFloat));
        float* a_ptr = x_fp32.mutable_data<float>();
        float16_t* b_ptr = param.x->mutable_data<float16_t>();
        fill_data_rand<float16_t>(b_ptr, -1.f, 1.f, param.x->numel());
        // fill_data_const<float16_t>(b_ptr, -1.f, param.x->numel());
        fp16_to_float(param.x->data<float16_t>(), a_ptr, param.x->numel());
        auto din = param.x->data<float16_t>();
        auto din_fp32 = x_fp32.data<float>();

        Tensor tout_basic;
        Tensor tout_basic_fp16;
        Tensor tout_basic_fp32;
        if (FLAGS_check_result) {
          tout_basic.set_precision(PRECISION(kFP16));
          tout_basic_fp16.set_precision(PRECISION(kFP16));
          tout_basic_fp32.set_precision(PRECISION(kFloat));
          tout_basic.Resize(dim_out);
          tout_basic_fp16.Resize(dim_out);
          tout_basic_fp32.Resize(dim_out);
          fill_tensor_const(tout_basic_fp32, 0.f);
          auto dout_basic = tout_basic.mutable_data<float16_t>();
          auto dout_basic_fp32 = tout_basic_fp32.mutable_data<float>();
          fill_data_const<float16_t>(dout_basic, 0.f, tout_basic.numel());
          pooling_basic<float, float>(din_fp32,
                                      dout_basic_fp32,
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
          // fp32 -> fp16
          auto dout_basic_fp16_ptr = tout_basic_fp16.mutable_data<float16_t>();
          float_to_fp16(
              dout_basic_fp32, dout_basic_fp16_ptr, tout_basic_fp16.numel());
        }
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
        print_gops_info("pool_fp16", dim_in, dim_out, t0, gops);
        if (FLAGS_check_result) {
          double max_ratio = 0;
          double max_diff = 0;
          auto basic_ptr = tout_basic_fp16.data<float16_t>();
          auto saber_ptr = param.output->data<float16_t>();
          Tensor tdiff;
          tdiff.Resize(tout_basic.dims());
          tdiff.set_precision(PRECISION(kFP16));
          auto ptr = tdiff.mutable_data<float16_t>();

          data_diff(basic_ptr,
                    saber_ptr,
                    ptr,
                    tout_basic.numel(),
                    max_ratio,
                    max_diff);
          print_diff_info(max_diff, max_ratio);
          if (std::abs(max_ratio) > 1e-3f) {
            if (max_diff > 4e-3f) {
              int64_t size = tout_basic.numel();
              int64_t width = tout_basic.dims()[tout_basic.dims().size() - 1];
              print_tensor_info_fp16(basic_ptr, saber_ptr, ptr, size, width);
              print_pool_success_or_fail_info("pool_fp16",
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
        print_pool_success_or_fail_info("pool_fp16",
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
void test_pool_fp16(const std::vector<DDim>& input_dims,
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
                            for (auto& batch : {1}) {
                              for (auto& h :
                                   {1, 2, 3, 4, 11, 15, 19, 31, 32, 28}) {
                                dims.push_back(DDim({batch, cin, h, h}));
                              }
                            }
                            test_pool_fp16(
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

#ifdef LITE_WITH_ARM8_SVE2  /// global_pool
TEST(TesPoolGlobal, test_pool_fp16_global) {
  for (auto& h : {51})
    test_pool_fp16({DDim({1, 64, h, h})},
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

TEST(TesPoolBasicAdaptive, test_pool_fp16_adaptive_size) {
  test_pool_fp16(
      {DDim({4, 32, 80, 80}), DDim({4, 32, 32, 32}), DDim({4, 32, 16, 16})},
      {4, 4},
      {5, 5},
      {0, 0, 0, 0},
      FLAGS_ceil_mode,
      false,
      true,
      true,
      false,
      "avg",
      {1},
      {0},
      {DDim({4, 32, 4, 4})});
  test_pool_fp16(
      {DDim({1, 3, 80, 80}), DDim({1, 3, 32, 32}), DDim({1, 3, 16, 16})},
      {4, 4},
      {5, 5},
      {0, 0, 0, 0},
      FLAGS_ceil_mode,
      false,
      true,
      true,
      false,
      "avg",
      {1},
      {0},
      {DDim({1, 3, 2, 2})});
}

TEST(TesPoolCustom, test_pool_fp16_custom_size) {
  test_pool_fp16(
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
