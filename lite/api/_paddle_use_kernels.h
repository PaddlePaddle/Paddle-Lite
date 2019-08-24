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

/*
 * ATTENTION this header file can only include in .cc file.
 */

#pragma once
#include "paddle_lite_factory_helper.h"  // NOLINT
#ifndef LITE_WITH_FPGA
USE_LITE_KERNEL(feed, kHost, kAny, kAny, def);
USE_LITE_KERNEL(fetch, kHost, kAny, kAny, def);
USE_LITE_KERNEL(flatten, kHost, kAny, kAny, def);
USE_LITE_KERNEL(flatten2, kHost, kAny, kAny, def);
#else
USE_LITE_KERNEL(feed, kFPGA, kFP16, kNHWC, def);
USE_LITE_KERNEL(fetch, kFPGA, kFP16, kNHWC, def);
#endif

// host kernels
USE_LITE_KERNEL(reshape, kHost, kAny, kAny, def);
USE_LITE_KERNEL(reshape2, kHost, kAny, kAny, def);

#ifdef LITE_WITH_ARM
USE_LITE_KERNEL(fc, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(matmul, kARM, kFloat, kNCHW, def);  // for x2paddle
USE_LITE_KERNEL(scale, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(softmax, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(lrn, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(decode_bboxes, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(box_coder, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(depthwise_conv2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_add, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_mul, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_max, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_add_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_mul_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_max_activation, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(split, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(dropout, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(concat, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(pool2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu6, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(transpose2, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(batch_norm, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(power, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(shuffle_channel, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(yolo_box, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(argmax, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(axpy, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(leaky_relu, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu_clipped, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(prelu, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(sigmoid, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(tanh, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(swish, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(log, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(exp, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d_transpose, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(pad2d, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(prior_box, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(density_prior_box, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(negative, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(crop, kARM, kFloat, kNCHW, def);

USE_LITE_KERNEL(norm, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_softmax, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(im2sequence, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(bilinear_interp, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(nearest_interp, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(logical_xor, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(logical_and, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(less_than, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(top_k, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(increment, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(write_to_array, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(read_from_array, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(multiclass_nms, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(reduce_max, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_expand, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(sequence_pool, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(shape, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(fill_constant, kARM, kFloat, kNCHW, def);
USE_LITE_KERNEL(cast, kARM, kFloat, kNCHW, def)
USE_LITE_KERNEL(slice, kARM, kFloat, kNCHW, def)
USE_LITE_KERNEL(squeeze, kARM, kFloat, kNCHW, def)   // for x2paddle
USE_LITE_KERNEL(squeeze2, kARM, kFloat, kNCHW, def)  // for x2paddle
USE_LITE_KERNEL(expand, kARM, kFloat, kNCHW, def)    // for x2paddle

USE_LITE_KERNEL(calib, kARM, kInt8, kNCHW, fp32_to_int8);
USE_LITE_KERNEL(calib, kARM, kInt8, kNCHW, int8_to_fp32);
USE_LITE_KERNEL(calib_once, kARM, kInt8, kNCHW, fp32_to_int8);
USE_LITE_KERNEL(calib_once, kARM, kInt8, kNCHW, int8_to_fp32);
USE_LITE_KERNEL(conv2d, kARM, kInt8, kNCHW, int8_out);
USE_LITE_KERNEL(conv2d, kARM, kInt8, kNCHW, fp32_out);
USE_LITE_KERNEL(fc, kARM, kInt8, kNCHW, int8out);
USE_LITE_KERNEL(fc, kARM, kInt8, kNCHW, fp32out);
USE_LITE_KERNEL(gru_unit, kARM, kFloat, kNCHW, def)
USE_LITE_KERNEL(gru, kARM, kFloat, kNCHW, def)
USE_LITE_KERNEL(beam_search_decode, kARM, kFloat, kNCHW, def)
USE_LITE_KERNEL(beam_search, kARM, kFloat, kNCHW, def)
USE_LITE_KERNEL(while, kARM, kFloat, kNCHW, def)
USE_LITE_KERNEL(lod_reset, kARM, kFloat, kNCHW, def)
USE_LITE_KERNEL(lookup_table, kARM, kFloat, kNCHW, def)
USE_LITE_KERNEL(is_empty, kARM, kFloat, kNCHW, def)
#endif

#ifdef LITE_WITH_X86
// NOTE all the X86 kernels are disabled temporarily for kernel are changed.
// USE_LITE_KERNEL(relu, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(mul, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(fc, kX86, kFloat, kNCHW, def);
USE_LITE_KERNEL(scale, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(fill_constant, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(square, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(elementwise_sub, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(elementwise_add, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(softmax, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(dropout, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(concat, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(conv2d, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(depthwise_conv2d, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(pool2d, kX86, kFloat, kNCHW, def);
// USE_LITE_KERNEL(batch_norm, kX86, kFloat, kNCHW, def);
#endif

#ifdef LITE_WITH_CUDA
USE_LITE_KERNEL(mul, kCUDA, kFloat, kNCHW, def);
USE_LITE_KERNEL(io_copy, kCUDA, kAny, kAny, host_to_device);
USE_LITE_KERNEL(io_copy, kCUDA, kAny, kAny, device_to_host);
USE_LITE_KERNEL(io_copy_once, kCUDA, kAny, kAny, host_to_device);
USE_LITE_KERNEL(io_copy_once, kCUDA, kAny, kAny, device_to_host);
#endif

#ifdef LITE_WITH_OPENCL
USE_LITE_KERNEL(io_copy, kOpenCL, kAny, kAny, host_to_device);
USE_LITE_KERNEL(io_copy, kOpenCL, kAny, kAny, device_to_host);
USE_LITE_KERNEL(io_copy_once, kOpenCL, kAny, kAny, host_to_device);
USE_LITE_KERNEL(io_copy_once, kOpenCL, kAny, kAny, device_to_host);

USE_LITE_KERNEL(fc, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(mul, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(elementwise_add, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(fusion_elementwise_add_activation, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(pool2d, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(relu, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(depthwise_conv2d, kOpenCL, kFloat, kNCHW, def);
USE_LITE_KERNEL(conv2d, kOpenCL, kFloat, kNCHW, def);
#endif

#ifdef LITE_WITH_NPU
USE_LITE_KERNEL(graph_op, kNPU, kFloat, kNCHW, def);
#endif
#ifdef LITE_WITH_FPGA
USE_LITE_KERNEL(relu, kFPGA, kFP16, kNHWC, def);
USE_LITE_KERNEL(conv2d, kFPGA, kFP16, kNHWC, def);
USE_LITE_KERNEL(elementwise_add, kFPGA, kFP16, kNHWC, def);
USE_LITE_KERNEL(fusion_elementwise_add_activation, kFPGA, kFP16, kNHWC, def);
USE_LITE_KERNEL(fc, kFPGA, kFP16, kNHWC, def);
USE_LITE_KERNEL(pool2d, kFPGA, kFP16, kNHWC, def);
USE_LITE_KERNEL(scale, kFPGA, kFP16, kNHWC, def);
USE_LITE_KERNEL(softmax, kFPGA, kFP16, kNHWC, def);
USE_LITE_KERNEL(io_copy, kFPGA, kAny, kAny, host_to_device);
USE_LITE_KERNEL(io_copy, kFPGA, kAny, kAny, device_to_host);
USE_LITE_KERNEL(io_copy_once, kFPGA, kAny, kAny, host_to_device_once);
USE_LITE_KERNEL(io_copy_once, kFPGA, kAny, kAny, device_to_host_once);
USE_LITE_KERNEL(calib, kFPGA, kFP16, kNHWC, fp32_to_fp16_fpga);
USE_LITE_KERNEL(calib, kFPGA, kFP16, kNHWC, fp16_to_fp32_fpga);
USE_LITE_KERNEL(calib_once, kFPGA, kFP16, kNHWC, fp32_to_fp16_fpga);
USE_LITE_KERNEL(calib_once, kFPGA, kFP16, kNHWC, fp16_to_fp32_fpga);
USE_LITE_KERNEL(layout, kFPGA, kAny, kNHWC, hwc_to_chw_fpga_fp16);
USE_LITE_KERNEL(layout, kFPGA, kAny, kNHWC, chw_to_hwc_fpga_fp16);
USE_LITE_KERNEL(layout_once, kFPGA, kAny, kNHWC, hwc_to_chw_fpga_fp16);
USE_LITE_KERNEL(layout_once, kFPGA, kAny, kNHWC, chw_to_hwc_fpga_fp16);
#endif
