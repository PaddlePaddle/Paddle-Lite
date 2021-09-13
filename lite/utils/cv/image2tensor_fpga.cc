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

#include "lite/utils/cv/image2tensor_fpga.h"

#include <memory.h>
#include <stdint.h>
#include <cmath>
#include <fstream>
#include <iostream>

#include "lite/utils/log/logging.h"

namespace paddle {
namespace lite {
namespace utils {
namespace cv {

void bgr_to_tensor_hwc(const uint8_t* src,
                       paddle::zynqmp::float16* output,
                       ImageFormat srcFormat,
                       ImageFormat dstFormat,
                       int srcw,
                       int srch,
                       int dstw,
                       int dsth,
                       float* means,
                       float* scales);

/*
  * change image data to tensor data
  * support image format is BGR(RGB), Data layout is NHWC a
  * param src: input image data
  * param dstTensor: output tensor data
  * param srcFormat: input image format, support BGR(GRB)
  * param dstFormat: input model format, support BGR(GRB)
  * param srcw: input image width
  * param srch: input image height
  * param dstw: input model width
  * param dsth: input model  height
  * param layout: output tensor layout，support NHWC
  * param means: means of image
  * param scales: scales of image
*/
void Image2TensorFpga::choose(const uint8_t* src,
                              Tensor* dst,
                              ImageFormat srcFormat,
                              ImageFormat dstFormat,
                              LayoutType layout,
                              int srcw,
                              int srch,
                              int dstw,
                              int dsth,
                              float* means,
                              float* scales) {
  paddle::zynqmp::float16* output =
      dst->mutable_data<paddle::zynqmp::float16>();
  if (layout == LayoutType::kNHWC && (srcFormat == BGR || srcFormat == RGB)) {
    impl_ = bgr_to_tensor_hwc;
  } else {
    printf("layout: %d or image format: %d is not supported! \n",
           static_cast<int>(layout),
           srcFormat);
    return;
  }
  impl_(
      src, output, srcFormat, dstFormat, srcw, srch, dstw, dsth, means, scales);
}

void bgr_to_tensor_hwc(const uint8_t* src,
                       paddle::zynqmp::float16* output,
                       ImageFormat srcFormat,
                       ImageFormat dstFormat,
                       int srcw,
                       int srch,
                       int dstw,
                       int dsth,
                       float* means,
                       float* scales) {
  int channel = 3;
  if (srcFormat == ImageFormat::BGR || srcFormat == ImageFormat::RGB) {
    channel = 3;
  } else if (srcFormat == ImageFormat::YUV422) {
    channel = 2;
  } else if (srcFormat == ImageFormat::YUV444) {
    LOG(FATAL) << "input image format YUV444 is not supported!";
    // (chonwhite) implement this;
  }
  int wc_align_size =
      paddle::zynqmp::align_to_x(srcw * channel, IMAGE_ALIGNMENT);

  int align_size = 1 * srch * wc_align_size;
  uint8_t* img = reinterpret_cast<uint8_t*>(
      paddle::zynqmp::fpga_malloc(align_size * sizeof(uint8_t)));
  memset(img, 0, align_size);

  int mod = wc_align_size - srcw * channel;
  int aligned_count = srcw * channel;
  uint8_t* src_uint8 = const_cast<uint8_t*>(src);

  for (int i = 0; i < srch; i++) {
    uint8_t* data = src_uint8 + i * (srcw * channel);
    uint8_t* dst = img + i * (srcw * channel + mod);
    memcpy(dst, data, aligned_count * sizeof(uint8_t));
  }

  paddle::zynqmp::PreprocessArgs preprocess_args = {0};

  preprocess_args.input_width = srcw;
  preprocess_args.input_height = srch;
  preprocess_args.output_width = dstw;
  preprocess_args.output_height = dsth;
  float height_ratio = srch * 1.0f / dsth;
  float width_ratio = srcw * 1.0f / dstw;
  preprocess_args.height_ratio = (uint32_t)(height_ratio * pow(2, 8));
  preprocess_args.width_ratio = (uint32_t)(width_ratio * pow(2, 8));
  preprocess_args.mean0 = paddle::zynqmp::fp32_2_fp16(means[0] * scales[0]);
  preprocess_args.mean1 = paddle::zynqmp::fp32_2_fp16(means[1] * scales[1]);
  preprocess_args.mean2 = paddle::zynqmp::fp32_2_fp16(means[2] * scales[2]);
  preprocess_args.scale0 = paddle::zynqmp::fp32_2_fp16(scales[0]);
  preprocess_args.scale1 = paddle::zynqmp::fp32_2_fp16(scales[1]);
  preprocess_args.scale2 = paddle::zynqmp::fp32_2_fp16(scales[2]);
  preprocess_args.rd_ring_buf_size = 1;
  preprocess_args.wr_ring_buf_size = 1;

  if (srcFormat == ImageFormat::BGR) {
    preprocess_args.vedio_in_fomat = 3;
  } else if (srcFormat == ImageFormat::RGB) {
    preprocess_args.vedio_in_fomat = 2;
  } else if (srcFormat == ImageFormat::YUV422) {
    preprocess_args.vedio_in_fomat = 0;
  } else if (srcFormat == ImageFormat::YUV444) {
    preprocess_args.vedio_in_fomat = 1;
  }

  if (dstFormat == ImageFormat::BGR) {
    preprocess_args.vedio_out_fomat = 1;
  } else if (dstFormat == ImageFormat::RGB) {
    preprocess_args.vedio_out_fomat = 0;
  }

  preprocess_args.vedio_source = 0;
  preprocess_args.mean_scale_enabled = 1;

  preprocess_args.input_image_address = img;
  preprocess_args.output_image_address = output;

  int in_size =
      srch * paddle::zynqmp::align_to_x(srcw * channel, IMAGE_ALIGNMENT);
  paddle::zynqmp::fpga_flush(preprocess_args.input_image_address,
                             in_size * sizeof(uint8_t));
  int ret = paddle::zynqmp::compute_preprocess(preprocess_args);
  if (ret == 0) {
    int output_size =
        dsth * paddle::zynqmp::align_to_x(dstw * 3, IMAGE_ALIGNMENT);
    paddle::zynqmp::fpga_invalidate(
        preprocess_args.output_image_address,
        output_size * sizeof(paddle::zynqmp::float16));
  }

  paddle::zynqmp::fpga_free(img);
}

}  // namespace cv
}  // namespace utils
}  // namespace lite
}  // namespace paddle
