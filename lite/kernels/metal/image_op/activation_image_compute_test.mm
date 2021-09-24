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

#include "lite/kernels/metal/image_op/activation_image_compute.h"
#include "lite/core/op_registry.h"
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace metal {

template <typename dtype>
void relu_compute_ref(const operators::ActivationParam& param) {
    DDim x_dims = param.X->dims();
    auto x_data = param.X->data<dtype>();
    auto y_data = param.Out->mutable_data<dtype>();

    int64_t outer_size = 0;
    int64_t channel_size = 0;
    int64_t inner_size = 0;

    outer_size = x_dims[0];
    channel_size = x_dims[1];
    inner_size = x_dims.Slice(2, x_dims.size()).production();

    auto x_ptr = const_cast<dtype*>(x_data);
    auto y_ptr = y_data;
    for (int o = 0; o < outer_size; o++) {
        for (int c = 0; c < channel_size; c++) {
            for (int i = 0; i < inner_size; i++) {
                *y_ptr = std::max<dtype>(*x_ptr, 0.0);
                ;
                x_ptr++;
                y_ptr++;
            }
        }
    }
}

TEST(relu_metal, retrive_op) {
    auto relu = KernelRegistry::Global().Create(
        "relu", TARGET(kMetal), PRECISION(kFloat), DATALAYOUT(kMetalTexture2DArray));

    ASSERT_FALSE(relu.empty());
    ASSERT_TRUE(relu.front());
}

TEST(relu_metal, init) {
    ActivationImageCompute<float, PRECISION(kFloat)> relu;
    ASSERT_EQ(relu.precision(), PRECISION(kFloat));
    ASSERT_EQ(relu.target(), TARGET(kMetal));
}

TEST(relu_metal, compute) {
    for (auto n : {1, 2}) {
        for (auto c : {/*6, 32 , */ 128}) {
            for (auto h : {/*9, 18 , 56 , 112, 224, */ 512}) {
                for (auto w : {18 /*, 56, 112, 224, 512*/}) {
                    for (auto is_test : {/*false, */ true}) {
                        for (auto use_global_stats : {false, true}) {
                            for (auto epsilon : {1e-4f, 1e-5f}) {
                                for (auto momentum : {0.9f, 0.99f}) {
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
                                        auto* y_data = y.mutable_data<float>();

                                        for (int i = 0; i < x.dims().production(); i++) {
                                            auto sign = pow(-1, i);
                                            x_data[i] = sign * static_cast<float>(i % 64);
                                        }

                                        auto x_dev_ptr = x_dev.mutable_data<float, MetalImage>(
                                            x_dev.dims(), {0, 2, 3, 1}, (void*)x_data);
                                        auto y_host_ptr = y.mutable_data<float>();

                                        {
                                            // judge the input
                                            Tensor x_from_dev;
                                            x_from_dev.Resize(in_out_shape);
                                            auto x_from_dev_ptr = x_from_dev.mutable_data<float>();
                                            x_dev_ptr->CopyToNCHW<float>(x_from_dev_ptr);
                                            for (int i = 0; i < x_from_dev.dims().production();
                                                 i++) {
                                                ASSERT_NEAR(x_from_dev_ptr[i], x_data[i], 1e-5);
                                            }
                                        }

                                        // prepare kernel params and run
                                        ActivationImageCompute<float, PRECISION(kFloat)> relu;
                                        std::unique_ptr<KernelContext> ctx(new KernelContext);
                                        ctx->As<ContextMetal>().InitOnce();

                                        auto mt = (MetalContext*)ctx->As<ContextMetal>().context();
                                        mt->set_metal_path(
                                            "/Users/liuzheyuan/code/Paddle-Lite/cmake-build-debug/"
                                            "lite/"
                                            "backends/metal/lite.metallib");

                                        relu.SetContext(std::move(ctx));
                                        operators::ActivationParam param;
                                        param.active_type = lite_api::ActivationType::kRelu;
                                        param.X = &x_dev;
                                        param.Out = &y_dev;
                                        relu.SetParam(param);
                                        relu.Launch();

                                        auto y_dev_ptr = y_dev.data<float, MetalImage>();
                                        y_dev_ptr->CopyToNCHW<float>(y_data);

                                        // invoking ref implementation and compare results
                                        param.X = &x;
                                        param.Out = &y_ref;
                                        relu_compute_ref<float>(param);
                                        auto* y_ref_data = y_ref.mutable_data<float>();

                                        for (int i = 0; i < y.dims().production(); i++) {
                                            ASSERT_NEAR(y_data[i], y_ref_data[i], 1e-5);
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

TEST(relu_metal_half, retrive_op_half) {
    auto relu_half = KernelRegistry::Global().Create(
        "relu", TARGET(kMetal), PRECISION(kFP16), DATALAYOUT(kMetalTexture2DArray));
    ASSERT_FALSE(relu_half.empty());
    ASSERT_TRUE(relu_half.front());
}

TEST(relu_metal_half, init) {
    ActivationImageCompute<MetalHalf, PRECISION(kFP16)> relu_half;
    ASSERT_EQ(relu_half.precision(), PRECISION(kFP16));
    ASSERT_EQ(relu_half.target(), TARGET(kMetal));
}

TEST(relu_metal_half, compute) {
    for (auto n : {1, 2}) {
        for (auto c : {/*6, 32 , */ 128}) {
            for (auto h : {/*9, 18 , 56 , 112, 224, */ 512}) {
                for (auto w : {18 /*, 56, 112, 224, 512*/}) {
                    for (auto is_test : {/*false, */ true}) {
                        for (auto use_global_stats : {false, true}) {
                            for (auto epsilon : {1e-4f, 1e-5f}) {
                                for (auto momentum : {0.9f, 0.99f}) {
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
                                        auto* y_data = y.mutable_data<float>();

                                        for (int i = 0; i < x.dims().production(); i++) {
                                            auto sign = pow(-1, i);
                                            x_data[i] = sign * static_cast<float>(i % 64);
                                        }

                                        //                    auto x_dev_ptr =
                                        //                    x_dev.mutable_data<float,
                                        //                    metal_image>(n,
                                        //                    c, h, w, (void*)x_data);
                                        auto x_dev_ptr = x_dev.mutable_data<MetalHalf, MetalImage>(
                                            x_dev.dims(), {0, 2, 3, 1});
                                        x_dev_ptr->CopyFromNCHW<float>(x_data);
                                        auto y_host_ptr = y.mutable_data<float>();

                                        {
                                            // judge the input
                                            Tensor x_from_dev;
                                            x_from_dev.Resize(in_out_shape);
                                            auto x_from_dev_ptr = x_from_dev.mutable_data<float>();
                                            x_dev_ptr->CopyToNCHW<float>(
                                                reinterpret_cast<float*>(x_from_dev_ptr));
                                            for (int i = 0; i < x_from_dev.dims().production();
                                                 i++) {
                                                ASSERT_NEAR(x_from_dev_ptr[i], x_data[i], 1e-5);
                                            }
                                        }

                                        // prepare kernel params and run
                                        ActivationImageCompute<MetalHalf, PRECISION(kFP16)> relu;
                                        std::unique_ptr<KernelContext> ctx(new KernelContext);
                                        ctx->As<ContextMetal>().InitOnce();
                                        relu.SetContext(std::move(ctx));
                                        operators::ActivationParam param;
                                        param.active_type = lite_api::ActivationType::kRelu;
                                        param.X = &x_dev;
                                        param.Out = &y_dev;
                                        relu.SetParam(param);
                                        relu.Launch();

                                        auto y_dev_ptr = y_dev.data<MetalHalf, MetalImage>();
                                        y_dev_ptr->CopyToNCHW<float>(y_data);

                                        // invoking ref implementation and compare results
                                        param.X = &x;
                                        param.Out = &y_ref;
                                        relu_compute_ref<float>(param);
                                        auto* y_ref_data = y_ref.mutable_data<float>();

                                        for (int i = 0; i < y.dims().production(); i++) {
                                            ASSERT_NEAR(y_data[i], y_ref_data[i], 1e-5);
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

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(relu, kMetal, kFloat, kMetalTexture2DArray, def);
USE_LITE_KERNEL(relu, kMetal, kFP16, kMetalTexture2DArray, def);
