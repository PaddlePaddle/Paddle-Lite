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

#ifndef LITE_BACKENDS_METAL_METAL_CONVERTER_H_
#define LITE_BACKENDS_METAL_METAL_CONVERTER_H_

#include "lite/backends/metal/metal_common.h"
#include "lite/backends/metal/metal_half.h"
#include "lite/core/dim.h"

namespace paddle {
namespace lite {

template <typename P>
class DataConverter {
   public:
    virtual void Convert(P* from, P* to, DDim fromDim) = 0;
    virtual DDim GetToDim(DDim fromDim) = 0;
    virtual int Capacity(DDim fromDim) {
        return 0;
    }
};

template <typename P>
class MPSPointerConverter : public DataConverter<P> {
    /// [ outputChannels ][ inputChannels ][ kernelHeight ][ kernelWidth ] ->
    /// [ outputChannels ][ kernelHeight ][ kernelWidth ][ inputChannels ]
    /// - Parameters:
    ///   - from: from pointer
    ///   - to: to pointer
    void Convert(P* from, P* to, DDim fromDim) override {
        auto outputChannels = fromDim[0];
        auto inputChannels = fromDim[1];
        auto kernelHeight = fromDim[2];
        auto kernelWidth = fromDim[3];

        for (int outChannel = 0; outChannel < outputChannels; outChannel++) {
            for (int kernelH = 0; kernelH < kernelHeight; kernelH++) {
                for (int kernelW = 0; kernelW < kernelWidth; kernelW++) {
                    for (int inChannel = 0; inChannel < inputChannels; inChannel++) {
                        to[outChannel * inputChannels * kernelHeight * kernelWidth +
                            kernelH * kernelWidth * inputChannels + kernelW * inputChannels +
                            inChannel] =
                            from[outChannel * inputChannels * kernelHeight * kernelWidth +
                                 inChannel * kernelHeight * kernelWidth + kernelH * kernelWidth +
                                 kernelW];
                    }
                }
            }
        }
    }

    DDim GetToDim(DDim fromDim) override {
        auto outputChannels = fromDim[0];
        auto inputChannels = fromDim[1];
        auto kernelHeight = fromDim[2];
        auto kernelWidth = fromDim[3];
        auto toDim = DDimLite({outputChannels, kernelHeight, kernelWidth, inputChannels});
        return toDim;
    }
};

template <typename P>
class ConvTransposeConverter : public DataConverter<P> {
    /// [ outputChannels ][ inputChannels ][ kernelHeight ][ kernelWidth ] ->
    /// [ outputChannels ][ kernelHeight ][ kernelWidth ][ inputChannels ]
    /// - Parameters:
    ///   - from: from pointer
    ///   - to: to pointer
    void Convert(P* from, P* to, DDim fromDim) override {
        auto N = fromDim[0];
        auto C = fromDim[1];
        auto H = fromDim[2];
        auto W = fromDim[3];
        assert(H == 3 || W == 3);
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                auto offset = n * C * H * W + c * H * W;
                to[offset] = from[offset + 8];
                to[offset + 1] = from[offset + 7];
                to[offset + 2] = from[offset + 6];
                to[offset + 3] = from[offset + 5];
                to[offset + 4] = from[offset + 4];
                to[offset + 5] = from[offset + 3];
                to[offset + 6] = from[offset + 2];
                to[offset + 7] = from[offset + 1];
                to[offset + 8] = from[offset];
            }
        }
    }

    DDim GetToDim(DDim fromDim) override {
        auto N = fromDim[0];
        auto C = fromDim[1];
        auto H = fromDim[2];
        auto W = fromDim[3];
        auto toDim = DDimLite({N, C, H, W});
        return toDim;
    }
};

template <typename P>
class WinogradPointerConverter : public DataConverter<P> {
    void Convert(P* from, P* to, DDim fromDim) override {
        auto N = fromDim[0];
        auto C = fromDim[1];
        auto H = fromDim[2];
        auto W = fromDim[3];
        if (H != 3 || W != 3) {
            throw std::logic_error("WinogradPointerConverter Convert H and W must equal to 3");
        }
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                auto fromOffset = n * C * H * W + c * H * W;
                auto toOffset = n * C * (H + 1) * (W + 1) + c * (H + 1) * (W + 1);

                auto f = [&](int h, int w) -> P { return from[fromOffset + h * W + w]; };
                auto c05 = (P)(0.5);
                auto c025 = (P)(0.25);
                to[toOffset] = f(0, 0);
                to[toOffset + 1] = c05 * f(0, 0);
                to[toOffset + 1] = to[toOffset + 1] + c05 * f(0, 1);
                to[toOffset + 1] = to[toOffset + 1] + c05 * f(0, 2);
                to[toOffset + 2] = c05 * f(0, 0);
                to[toOffset + 2] = to[toOffset + 2] - c05 * f(0, 1);
                to[toOffset + 2] = to[toOffset + 2] + c05 * f(0, 2);
                to[toOffset + 3] = f(0, 2);
                to[toOffset + 4] = c05 * f(0, 0);
                to[toOffset + 4] = to[toOffset + 4] + c05 * f(1, 0);
                to[toOffset + 4] = to[toOffset + 4] + c05 * f(2, 0);
                to[toOffset + 5] = c025 * f(0, 0);
                to[toOffset + 5] = to[toOffset + 5] + c025 * f(0, 1);
                to[toOffset + 5] = to[toOffset + 5] + c025 * f(0, 2);
                to[toOffset + 5] = to[toOffset + 5] + c025 * f(1, 0);
                to[toOffset + 5] = to[toOffset + 5] + c025 * f(1, 1);
                to[toOffset + 5] = to[toOffset + 5] + c025 * f(1, 2);
                to[toOffset + 5] = to[toOffset + 5] + c025 * f(2, 0);
                to[toOffset + 5] = to[toOffset + 5] + c025 * f(2, 1);
                to[toOffset + 5] = to[toOffset + 5] + c025 * f(2, 2);
                to[toOffset + 6] = c025 * f(0, 0);
                to[toOffset + 6] = to[toOffset + 6] - c025 * f(0, 1);
                to[toOffset + 6] = to[toOffset + 6] + c025 * f(0, 2);
                to[toOffset + 6] = to[toOffset + 6] + c025 * f(1, 0);
                to[toOffset + 6] = to[toOffset + 6] - c025 * f(1, 1);
                to[toOffset + 6] = to[toOffset + 6] + c025 * f(1, 2);
                to[toOffset + 6] = to[toOffset + 6] + c025 * f(2, 0);
                to[toOffset + 6] = to[toOffset + 6] - c025 * f(2, 1);
                to[toOffset + 6] = to[toOffset + 6] + c025 * f(2, 2);
                to[toOffset + 7] = c05 * f(0, 2);
                to[toOffset + 7] = to[toOffset + 7] + c05 * f(1, 2);
                to[toOffset + 7] = to[toOffset + 7] + c05 * f(2, 2);
                to[toOffset + 8] = c05 * f(0, 0);
                to[toOffset + 8] = to[toOffset + 8] - c05 * f(1, 0);
                to[toOffset + 8] = to[toOffset + 8] + c05 * f(2, 0);
                to[toOffset + 9] = c025 * f(0, 0);
                to[toOffset + 9] = to[toOffset + 9] + c025 * f(0, 1);
                to[toOffset + 9] = to[toOffset + 9] + c025 * f(0, 2);
                to[toOffset + 9] = to[toOffset + 9] - c025 * f(1, 0);
                to[toOffset + 9] = to[toOffset + 9] - c025 * f(1, 1);
                to[toOffset + 9] = to[toOffset + 9] - c025 * f(1, 2);
                to[toOffset + 9] = to[toOffset + 9] + c025 * f(2, 0);
                to[toOffset + 9] = to[toOffset + 9] + c025 * f(2, 1);
                to[toOffset + 9] = to[toOffset + 9] + c025 * f(2, 2);
                to[toOffset + 10] = c025 * f(0, 0);
                to[toOffset + 10] = to[toOffset + 10] - c025 * f(0, 1);
                to[toOffset + 10] = to[toOffset + 10] + c025 * f(0, 2);
                to[toOffset + 10] = to[toOffset + 10] - c025 * f(1, 0);
                to[toOffset + 10] = to[toOffset + 10] + c025 * f(1, 1);
                to[toOffset + 10] = to[toOffset + 10] - c025 * f(1, 2);
                to[toOffset + 10] = to[toOffset + 10] + c025 * f(2, 0);
                to[toOffset + 10] = to[toOffset + 10] - c025 * f(2, 1);
                to[toOffset + 10] = to[toOffset + 10] + c025 * f(2, 2);
                to[toOffset + 11] = c05 * f(0, 2);
                to[toOffset + 11] = to[toOffset + 11] - c05 * f(1, 2);
                to[toOffset + 11] = to[toOffset + 11] + c05 * f(2, 2);
                to[toOffset + 12] = f(2, 0);
                to[toOffset + 13] = c05 * f(2, 0);
                to[toOffset + 13] = to[toOffset + 13] + c05 * f(2, 1);
                to[toOffset + 13] = to[toOffset + 13] + c05 * f(2, 2);
                to[toOffset + 14] = c05 * f(2, 0);
                to[toOffset + 14] = to[toOffset + 14] - c05 * f(2, 1);
                to[toOffset + 14] = to[toOffset + 14] + c05 * f(2, 2);
                to[toOffset + 15] = f(2, 2);
            }
        }
    }

    DDim GetToDim(DDim fromDim) override {
        auto N = fromDim[0];
        auto C = fromDim[1];
        auto H = fromDim[2];
        auto W = fromDim[3];
        if (H != 3 || W != 3) {
            throw std::logic_error(
                "ERROR: WinogradPointerConverter GetToDim H and W must equal to 3");
        }
        auto toDim = DDimLite({N, C, H + 1, W + 1});
        return toDim;
    }

    int Capacity(DDim fromDim) override {
        auto N = fromDim[0];
        auto C = fromDim[1];
        auto H = fromDim[2];
        auto W = fromDim[3];
        if (H != 3 || W != 3) {
            throw std::logic_error(
                "ERROR: WinogradPointerConverter Capacity H and W must equal to 3");
        }
        return static_cast<int>(N * C * (H + 1) * (W + 1));
    }
};

class MetalConverter {
   public:
    template <typename SP, typename DP>
    static void NCHW2NHWC(DP* dstPtr, const SP* srcPtr, int N, int C, int H, int W) {
    }

    template <>
    void NCHW2NHWC<float, MetalHalf>(MetalHalf* dstPtr,
        const float* srcPtr,
        int N,
        int C,
        int H,
        int W) {
        auto HXW = H * W;
        auto CXHXW = C * H * W;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C; ++c) {
                        dstPtr[index] = MetalFloat2Half(srcPtr[n * CXHXW + c * HXW + h * W + w]);
                        index += 1;
                    }
                }
            }
        }
    }

    template <>
    void NCHW2NHWC<float, float>(float* dstPtr, const float* srcPtr, int N, int C, int H, int W) {
        auto HXW = H * W;
        auto CXHXW = C * H * W;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C; ++c) {
                        dstPtr[index] = (srcPtr[n * CXHXW + c * HXW + h * W + w]);
                        index += 1;
                    }
                }
            }
        }
    }

    template <>
    void NCHW2NHWC<MetalHalf, MetalHalf>(MetalHalf* dstPtr,
        const MetalHalf* srcPtr,
        int N,
        int C,
        int H,
        int W) {
        auto HXW = H * W;
        auto CXHXW = C * H * W;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    for (int c = 0; c < C; ++c) {
                        dstPtr[index] = (srcPtr[n * CXHXW + c * HXW + h * W + w]);
                        index += 1;
                    }
                }
            }
        }
    }

    template <typename DP, typename SP>
    static void NHWCExpand2NCHW(DP* dstPtr, const SP* srcPtr, int N, int C, int H, int W) {
        auto C_EXPAND = ((C + 3) / 4) * 4;
        auto HXWXC_E = H * W * C_EXPAND;
        auto WXC_E = W * C_EXPAND;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        dstPtr[index] = srcPtr[n * HXWXC_E + h * WXC_E + w * C_EXPAND + c];
                        index += 1;
                    }
                }
            }
        }
    }

    template <>
    void NHWCExpand2NCHW<float, MetalHalf>(float* dstPtr,
        const MetalHalf* srcPtr,
        int N,
        int C,
        int H,
        int W) {
        auto C_EXPAND = ((C + 3) / 4) * 4;
        auto HXWXC_E = H * W * C_EXPAND;
        auto WXC_E = W * C_EXPAND;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        dstPtr[index] =
                            MetalHalf2Float(srcPtr[n * HXWXC_E + h * WXC_E + w * C_EXPAND + c]);
                        index += 1;
                    }
                }
            }
        }
    }

    template <>
    void NHWCExpand2NCHW<float, float>(float* dstPtr,
        const float* srcPtr,
        int N,
        int C,
        int H,
        int W) {
        auto C_EXPAND = ((C + 3) / 4) * 4;
        auto HXWXC_E = H * W * C_EXPAND;
        auto WXC_E = W * C_EXPAND;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        dstPtr[index] = srcPtr[n * HXWXC_E + h * WXC_E + w * C_EXPAND + c];
                        index += 1;
                    }
                }
            }
        }
    }

    template <>
    void NHWCExpand2NCHW<MetalHalf, float>(MetalHalf* dstPtr,
        const float* srcPtr,
        int N,
        int C,
        int H,
        int W) {
        auto C_EXPAND = ((C + 3) / 4) * 4;
        auto HXWXC_E = H * W * C_EXPAND;
        auto WXC_E = W * C_EXPAND;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        dstPtr[index] =
                            MetalFloat2Half(srcPtr[n * HXWXC_E + h * WXC_E + w * C_EXPAND + c]);
                        index += 1;
                    }
                }
            }
        }
    }

    template <>
    void NHWCExpand2NCHW<MetalHalf, MetalHalf>(MetalHalf* dstPtr,
        const MetalHalf* srcPtr,
        int N,
        int C,
        int H,
        int W) {
        auto C_EXPAND = ((C + 3) / 4) * 4;
        auto HXWXC_E = H * W * C_EXPAND;
        auto WXC_E = W * C_EXPAND;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        dstPtr[index] = (srcPtr[n * HXWXC_E + h * WXC_E + w * C_EXPAND + c]);
                        index += 1;
                    }
                }
            }
        }
    }

    template <typename DP, typename SP>
    static void NHWC2NCHW(DP* dstPtr, const SP* srcPtr, int N, int C, int H, int W) {
        auto C_EXPAND = C;
        auto HXWXC_E = H * W * C_EXPAND;
        auto WXC_E = W * C_EXPAND;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        dstPtr[index] = srcPtr[n * HXWXC_E + h * WXC_E + w * C_EXPAND + c];
                        index += 1;
                    }
                }
            }
        }
    }

    template <>
    void NHWC2NCHW<float, float>(float* dstPtr, const float* srcPtr, int N, int C, int H, int W) {
        auto C_EXPAND = C;
        auto HXWXC_E = H * W * C_EXPAND;
        auto WXC_E = W * C_EXPAND;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        dstPtr[index] = srcPtr[n * HXWXC_E + h * WXC_E + w * C_EXPAND + c];
                        index += 1;
                    }
                }
            }
        }
    }

    template <>
    void NHWC2NCHW<float, MetalHalf>(float* dstPtr,
        const MetalHalf* srcPtr,
        int N,
        int C,
        int H,
        int W) {
        auto C_EXPAND = C;
        auto HXWXC_E = H * W * C_EXPAND;
        auto WXC_E = W * C_EXPAND;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        dstPtr[index] =
                            MetalHalf2Float(srcPtr[n * HXWXC_E + h * WXC_E + w * C_EXPAND + c]);
                        index += 1;
                    }
                }
            }
        }
    }

    template <>
    void NHWC2NCHW<MetalHalf, MetalHalf>(MetalHalf* dstPtr,
        const MetalHalf* srcPtr,
        int N,
        int C,
        int H,
        int W) {
        auto C_EXPAND = C;
        auto HXWXC_E = H * W * C_EXPAND;
        auto WXC_E = W * C_EXPAND;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        dstPtr[index] = (srcPtr[n * HXWXC_E + h * WXC_E + w * C_EXPAND + c]);
                        index += 1;
                    }
                }
            }
        }
    }

    template <>
    void NHWC2NCHW<MetalHalf, float>(MetalHalf* dstPtr,
        const float* srcPtr,
        int N,
        int C,
        int H,
        int W) {
        auto C_EXPAND = C;
        auto HXWXC_E = H * W * C_EXPAND;
        auto WXC_E = W * C_EXPAND;
        int index = 0;
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        dstPtr[index] =
                            MetalFloat2Half(srcPtr[n * HXWXC_E + h * WXC_E + w * C_EXPAND + c]);
                        index += 1;
                    }
                }
            }
        }
    }
};

}  // namespace lite
}  // namespace paddle
#endif  // LITE_BACKENDS_METAL_METAL_CONVERTER_H_
