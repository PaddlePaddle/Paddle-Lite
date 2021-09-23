// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/metal/image_op/batch_norm_image_compute.h"
#include "lite/core/op_registry.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename dtype>
void batch_norm_compute_ref(const operators::BatchNormParam& param) {
    DDim x_dims = param.x->dims();
    auto x_data = param.x->mutable_data<dtype>();
    auto scale_data = param.scale->mutable_data<dtype>();
    auto bias_data = param.bias->mutable_data<dtype>();
    auto mean_data = param.mean->mutable_data<dtype>();
    auto variance_data = param.variance->mutable_data<dtype>();
    auto y_data = param.y->mutable_data<dtype>();
    float epsilon = param.epsilon;
    float momentum = param.momentum;
    DataLayoutType data_layout = param.data_layout;

    bool global_stats = param.is_test || param.use_global_stats;
    if (global_stats) {
        int64_t outer_size = 0;
        int64_t channel_size = 0;
        int64_t inner_size = 0;
        switch (data_layout) {
            case DATALAYOUT(kNCHW):
                outer_size = x_dims[0];
                channel_size = x_dims[1];
                inner_size = x_dims.Slice(2, x_dims.size()).production();
                break;
            default:
                LOG(FATAL) << "Unknown storage order: " << DataLayoutToStr(data_layout);
                break;
        }

        for (int c = 0; c < channel_size; c++) {
            auto inv_std = (float)(1.0f / std::sqrt(variance_data[c] + param.epsilon));
            bias_data[c] = bias_data[c] - mean_data[c] * inv_std * scale_data[c];
            scale_data[c] = inv_std * scale_data[c];
#if 0
          std::cout << "bias and scale out"<< std::endl;
          std::cout << bias_data[c] << " " << scale_data[c] << std::endl;
          std::cout << "end"<< std::endl;
#endif
        }

        auto x_ptr = x_data;
        auto y_ptr = y_data;
        for (int o = 0; o < outer_size; o++) {
            for (int c = 0; c < channel_size; c++) {
                for (int i = 0; i < inner_size; i++) {
                    *y_ptr = *x_ptr * scale_data[c] + bias_data[c];
                    x_ptr++;
                    y_ptr++;
                }
            }
        }
    } else {
        // TODO(hong19860320) calculate mean_out, variance_out, saved_mean and
        // saved_variance
    }
}

TEST(batch_norm_metal, retrive_op) {
    auto batch_norm = KernelRegistry::Global().Create(
        "batch_norm", TARGET(kMetal), PRECISION(kFloat), DATALAYOUT(kMetalTexture2DArray));
    ASSERT_FALSE(batch_norm.empty());
    ASSERT_TRUE(batch_norm.front());
}

TEST(batch_norm_metal, init) {
    BatchNormImageCompute<float, PRECISION(kFloat)> batch_norm;
    ASSERT_EQ(batch_norm.precision(), PRECISION(kFloat));
    ASSERT_EQ(batch_norm.target(), TARGET(kMetal));
}

TEST(batch_norm_metal, compute) {
    for (auto n : {1 /*,  2*/}) {
        for (auto c : {/*6,  32 , */ 128}) {
            for (auto h : {/*9,  18 , 56 , 112, 224,*/ 512}) {
                for (auto w : {18 /*, 56, 112, 224, 512*/}) {
                    for (auto is_test : {/*false, */ true}) {
                        for (auto use_global_stats : {/*false, */ true}) {
                            for (auto epsilon : {/*1e-4f,*/ 1e-5f}) {
                                for (auto momentum : {/*0.9f,*/ 0.99f}) {
                                    for (auto data_layout :
                                        {DATALAYOUT(kNCHW) /*, DATALAYOUT(kNHWC)*/}) {
                                        Tensor x;
                                        Tensor x_dev;
                                        Tensor scale;
                                        Tensor bias;
                                        Tensor mean;
                                        Tensor variance;
                                        Tensor y;
                                        Tensor y_dev;
                                        Tensor mean_out;
                                        Tensor variance_out;
                                        Tensor saved_mean;
                                        Tensor saved_variance;
                                        Tensor y_ref;
                                        Tensor mean_out_ref;
                                        Tensor variance_out_ref;
                                        Tensor saved_mean_ref;
                                        Tensor saved_variance_ref;
                                        // set the dims of input, output, ref output tensors
                                        std::vector<int64_t> in_out_shape;
                                        switch (data_layout) {
                                            case DATALAYOUT(kNCHW):
                                                in_out_shape = {n, c, h, w};
                                                break;
                                            default:
                                                LOG(FATAL) << "Unknown storage order: "
                                                           << DataLayoutToStr(data_layout);
                                                break;
                                        }
                                        x.Resize(in_out_shape);
                                        x_dev.Resize(in_out_shape);
                                        scale.Resize({c});
                                        bias.Resize({c});
                                        mean.Resize({c});
                                        variance.Resize({c});
                                        y.Resize(in_out_shape);
                                        y_dev.Resize(in_out_shape);
                                        mean_out.Resize({c});
                                        variance_out.Resize({c});
                                        saved_mean.Resize({c});
                                        saved_variance.Resize({c});
                                        y_ref.Resize(in_out_shape);
                                        mean_out_ref.Resize({c});
                                        variance_out_ref.Resize({c});
                                        saved_mean_ref.Resize({c});
                                        saved_variance_ref.Resize({c});

                                        // initialize the data of input tensors
                                        auto* x_data = x.mutable_data<float>();
                                        auto* scale_data = scale.mutable_data<float>();
                                        auto* bias_data = bias.mutable_data<float>();
                                        auto* mean_data = mean.mutable_data<float>();
                                        auto* variance_data = variance.mutable_data<float>();
                                        auto* y_data = y.mutable_data<float>();
                                        for (int i = 0; i < x.dims().production(); i++) {
                                            x_data[i] = static_cast<float>(i % 64);
                                        }
                                        for (int i = 0; i < scale.dims().production(); i++) {
                                            scale_data[i] = static_cast<float>(i) * 0.01f + 0.03f;
                                        }
                                        for (int i = 0; i < bias.dims().production(); i++) {
                                            bias_data[i] = static_cast<float>(i) * 0.065f + 0.1f;
                                        }
                                        for (int i = 0; i < mean.dims().production(); i++) {
                                            mean_data[i] = static_cast<float>(i) * 0.0565f;
                                        }
                                        for (int i = 0; i < variance.dims().production(); i++) {
                                            variance_data[i] = static_cast<float>(i) * 2.08f + 1.5f;
                                        }

                                        auto x_dev_ptr = x_dev.mutable_data<float, MetalImage>(
                                            x.dims(), {0, 2, 3, 1}, (void*)x_data);
                                        auto y_host_ptr = y.mutable_data<float>();

                                        if (false) {
                                            // judge the input
                                            Tensor x_from_dev;
                                            x_from_dev.Resize(in_out_shape);
                                            auto x_from_dev_ptr = x_from_dev.mutable_data<float>();
                                            x_dev_ptr->CopyToNCHW<float>(x_from_dev_ptr);
                                            for (int i = 0; i < x_from_dev.dims().production();
                                                 i++) {
                                                EXPECT_NEAR(x_from_dev_ptr[i], x_data[i], 1e-5);
                                            }
                                        }

                                        // prepare kernel params and run
                                        BatchNormImageCompute<float, PRECISION(kFloat)> batch_norm;
                                        std::unique_ptr<KernelContext> ctx(new KernelContext);
                                        ctx->As<ContextMetal>().InitOnce();
                                        auto mt = (MetalContext*)ctx->As<ContextMetal>().context();
                                        mt->set_metal_path(
                                            "/Users/liuzheyuan/code1/Paddle-Lite/cmake-build-debug/"
                                            "lite/"
                                            "backends/metal/lite.metallib");
                                        mt->CreateCommandBuffer();
                                        batch_norm.SetContext(std::move(ctx));
                                        operators::BatchNormParam param;

                                        param.x = &x_dev;
                                        param.scale = &scale;
                                        param.bias = &bias;
                                        param.mean = &mean;
                                        param.variance = &variance;
                                        param.is_test = is_test;
                                        param.use_global_stats = use_global_stats;
                                        param.epsilon = epsilon;
                                        param.momentum = momentum;
                                        param.data_layout = data_layout;
                                        param.y = &y_dev;
                                        param.mean_out = &mean_out;
                                        param.variance_out = &variance_out;
                                        param.saved_mean = &saved_mean;
                                        param.saved_variance = &saved_variance;
                                        batch_norm.SetParam(param);
                                        batch_norm.Launch();
                                        mt->WaitUntilCompleted();

                                        auto y_dev_ptr = y_dev.data<float, MetalImage>();

                                        y_dev_ptr->CopyToNCHW<float>(y_data);

                                        param.x = &x;
                                        param.scale = &scale;
                                        param.bias = &bias;
                                        param.mean = &mean;
                                        param.variance = &variance;
                                        param.is_test = is_test;
                                        param.use_global_stats = use_global_stats;
                                        param.epsilon = epsilon;
                                        param.momentum = momentum;
                                        param.data_layout = data_layout;
                                        param.y = &y_ref;
                                        param.mean_out = &mean_out_ref;
                                        param.variance_out = &variance_out_ref;
                                        param.saved_mean = &saved_mean_ref;
                                        param.saved_variance = &saved_variance_ref;

                                        batch_norm_compute_ref<float>(param);
                                        auto* y_ref_data = y_ref.mutable_data<float>();

                                        for (int i = 0; i < y.dims().production(); i++) {
                                            if (true)
                                                //                      if (std::abs(y_data[i] -
                                                //                      y_ref_data[i]) > 1e-5)if
                                                //                      (std::abs(y_data[i] -
                                                //                      y_ref_data[i]) > 1e-5) if(i>
                                                //                      103678)
                                                std::cout << "[" << n << "] "
                                                          << "[" << c << "] "
                                                          << "[" << h << "] "
                                                          << "[" << w << "] "
                                                          << "[" << i << "] " << y_data[i] << " : "
                                                          << y_ref_data[i] << std::endl;
                                            //                      ASSERT_NEAR(y_data[i],
                                            //                      y_ref_data[i], 1e-5);
                                        }

                                        std::cout << std::endl;
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

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(batch_norm, kMetal, kFloat, kMetalTexture2DArray, def);
USE_LITE_KERNEL(batch_norm, kMetal, kFP16, kMetalTexture2DArray, def);
