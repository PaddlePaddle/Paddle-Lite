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

#include "imgdnn_manager.h"  // NOLINT
#include <utility>

namespace paddle {
namespace lite {
namespace imagination_nna {

static void err_callback(imgdnn_report_flags flags,
                         const char **tensor_names,
                         int num_tensor_names,
                         imgdnn_err_code error_code,
                         const char *error_message) {
  std::string msg_prefix;
  switch (flags) {
    case imgdnn_report_flags::IMGDNN_REPORT_ERROR:
      msg_prefix = "ERROR";
      break;
    case imgdnn_report_flags::IMGDNN_REPORT_VERBOSE:
      msg_prefix = "VERBOSE";
      break;
    case imgdnn_report_flags::IMGDNN_REPORT_INFO:
      msg_prefix = "INFO";
      break;
    case imgdnn_report_flags::IMGDNN_REPORT_WARNING:
      msg_prefix = "WARNING";
      break;
    default:
      std::cerr << "unknown report flag in error callback" << std::endl;
  }

  std::cerr << msg_prefix << ": " << error_message << std::endl;
}

ImgdnnManager::ImgdnnManager() {
  err_ = imgdnnSetErrorHandler(err_callback);
  net_ = imgdnnCreateNetwork(&err_);
  ASSERT(err_ != IMGDNN_SUCCESS, "CreateNetwork failed!");

  unsigned int num_devices;
  err_ = imgdnnGetDevices(
      IMGDNN_DEVICE_TYPE_ACCELERATOR, 1, &device_, &num_devices);
  ASSERT(err_ != IMGDNN_SUCCESS, "GetDevices failed!");
  context_ = imgdnnCreateContext(num_devices, &device_, 0, &err_);
  ASSERT(err_ != IMGDNN_SUCCESS, "CreateContext failed!");
  binding_ = imgdnnCreateBinding(&err_);
  ASSERT(err_ != IMGDNN_SUCCESS, "CreateBinding failed!");
}

imgdnn_tensor ImgdnnManager::createConvolutionLayer(
    imgdnn_tensor input_tensor,
    imgdnn_tensor weights_tensor,
    imgdnn_tensor bias_tensor,
    imgdnn_quant_param dst_quant_param,
    unsigned int stride[2],
    unsigned int pad_begin[2],
    unsigned int pad_end[2],
    unsigned int dilation[2],
    bool use_dwconv) {
  imgdnn_tensor convw_tensor;
  if (use_dwconv) {
    // transpose weight
    int order[4] = {1, 0, 2, 3};
    imgdnn_tensor transport_weights =
        imgdnnNetworkTransposeOp(net_, weights_tensor, order, &err_);
    convw_tensor = imgdnnNetworkDepthConvolution2dOp_v2(net_,
                                                        input_tensor,
                                                        transport_weights,
                                                        stride,
                                                        pad_begin,
                                                        pad_end,
                                                        dilation,
                                                        &err_);
  } else {
    convw_tensor = imgdnnNetworkConvolution2dOp_v2(net_,
                                                   input_tensor,
                                                   weights_tensor,
                                                   stride,
                                                   pad_begin,
                                                   pad_end,
                                                   dilation,
                                                   &err_);
  }

  // debug
  imgdnn_tensor_descriptor desc_1;
  imgdnnGetTensorDescriptor(input_tensor, &desc_1);
  imgdnnGetTensorDescriptor(weights_tensor, &desc_1);
  imgdnnGetTensorDescriptor(convw_tensor, &desc_1);

  imgdnn_tensor conv2d_tensor;
  if (bias_tensor) {
    imgdnn_tensor convw_int_tensor = imgdnnNetworkCastOp(
        net_, convw_tensor, IMGDNN_TYPE_I32, nullptr, &err_);

    imgdnn_tensor_descriptor bias_desc;
    imgdnnGetTensorDescriptor(convw_tensor, &bias_desc);

    imgdnn_tensor broadcast2_tensor;
    broadcast2_tensor = imgdnnNetworkBroadcastOp(
        net_, bias_tensor, 2, bias_desc.size[2], &err_);

    imgdnn_tensor broadcast3_tensor;
    broadcast3_tensor = imgdnnNetworkBroadcastOp(
        net_, broadcast2_tensor, 3, bias_desc.size[3], &err_);

    conv2d_tensor = imgdnnNetworkBinaryOp(
        net_, convw_int_tensor, broadcast3_tensor, IMGDNN_OPERATION_ADD, &err_);
  } else {
    conv2d_tensor = convw_tensor;
  }

  imgdnn_tensor conv2d_out_tensor;
  imgdnn_tensor_descriptor desc;
  imgdnnGetTensorDescriptor(input_tensor, &desc);
  if (desc.type == IMGDNN_TYPE_Q_I8 || desc.type == IMGDNN_TYPE_Q_U8) {
    conv2d_out_tensor = imgdnnNetworkCastOp(
        net_, conv2d_tensor, desc.type, &dst_quant_param, &err_);
  }

  return conv2d_out_tensor;
}

imgdnn_tensor ImgdnnManager::createBatchNormLayer(imgdnn_tensor input_tensor,
                                                  const void *const avg_in,
                                                  const void *const var_in,
                                                  const float eps) {
  imgdnn_tensor bna_tensor;
  imgdnn_tensor average_tensor;
  imgdnn_tensor_descriptor av_desc;

  imgdnn_tensor broadcast2_tensor;
  imgdnn_tensor broadcast3_tensor;

  unsigned int buffer_size;

  imgdnn_tensor_descriptor in_desc;
  imgdnnGetTensorDescriptor(input_tensor, &in_desc);

  av_desc.dimensions = 2;
  av_desc.type = in_desc.type;
  av_desc.size[0] = in_desc.size[0];
  av_desc.size[1] = in_desc.size[1];

  average_tensor = createFixedInputTensor(&av_desc, avg_in, true);

  broadcast2_tensor =
      imgdnnNetworkBroadcastOp(net_, average_tensor, 2, in_desc.size[2], &err_);

  broadcast3_tensor = imgdnnNetworkBroadcastOp(
      net_, broadcast2_tensor, 3, in_desc.size[3], &err_);

  bna_tensor = imgdnnNetworkBinaryOp(
      net_, input_tensor, broadcast3_tensor, IMGDNN_OPERATION_SUB, &err_);

  imgdnn_tensor variance_tensor;
  imgdnn_tensor_descriptor va_desc;

  va_desc.dimensions = 2;
  va_desc.type = in_desc.type;
  va_desc.size[0] = in_desc.size[0];
  va_desc.size[1] = in_desc.size[1];

  buffer_size = imgdnnGetDescriptorSize(&va_desc, &err_);
  float *variance = reinterpret_cast<float *>(GetBufromPool(buffer_size));
  memcpy(variance, var_in, buffer_size);
  // Perform 1/sqrt(var+eps) and Update var_data.
  buffer_size /= sizeof(float);
  for (size_t i = 0; i < buffer_size; i++) {
    variance[i] = 1.0 / (sqrt(variance[i] + eps));
  }
  variance_tensor = createFixedInputTensor(&va_desc, variance, false);

  broadcast2_tensor = imgdnnNetworkBroadcastOp(
      net_, variance_tensor, 2, in_desc.size[2], &err_);

  broadcast3_tensor = imgdnnNetworkBroadcastOp(
      net_, broadcast2_tensor, 3, in_desc.size[3], &err_);

  imgdnn_tensor bn_tensor;
  bn_tensor = imgdnnNetworkBinaryOp(
      net_, bna_tensor, broadcast3_tensor, IMGDNN_OPERATION_MUL, &err_);

  return bn_tensor;
}

imgdnn_tensor ImgdnnManager::createPoolingLayer(
    imgdnn_tensor in_tensor,
    imgdnn_quant_param dst_quant_param,
    const unsigned int size[2],
    const unsigned int stride[2],
    const unsigned int pad_to_begin[2],
    const unsigned int pad_to_end[2],
    imgdnn_pooling_type type) {
  // debug
  imgdnn_tensor_descriptor desc_1;
  imgdnnGetTensorDescriptor(in_tensor, &desc_1);

  imgdnn_tensor pool_tensor = imgdnnNetworkPooling2dOp_v2(
      net_, in_tensor, size, stride, pad_to_begin, pad_to_end, type, &err_);
  // debug
  imgdnnGetTensorDescriptor(pool_tensor, &desc_1);

  imgdnn_tensor_descriptor desc;
  imgdnnGetTensorDescriptor(in_tensor, &desc);
  if (desc.type == IMGDNN_TYPE_Q_I8 || desc.type == IMGDNN_TYPE_Q_U8) {
    pool_tensor = imgdnnNetworkCastOp(
        net_, pool_tensor, desc.type, &dst_quant_param, &err_);
  }

  return pool_tensor;
}

imgdnn_tensor ImgdnnManager::createFullyConnectedLayer(
    imgdnn_tensor input_tensor,
    imgdnn_tensor weights_tensor,
    imgdnn_tensor bias_tensor,
    imgdnn_quant_param dst_quant_param) {
  imgdnn_tensor fcw_tensor;
  imgdnn_tensor fcb_tensor;

  imgdnn_tensor_descriptor in_desc;
  imgdnnGetTensorDescriptor(input_tensor, &in_desc);

  // int flatten_dim = 1
  for (unsigned i = 2; i < in_desc.dimensions; ++i)
    in_desc.size[1] *= in_desc.size[i];
  in_desc.dimensions = 2;

  auto reshaped_input =
      imgdnnNetworkReshapeOp(net_, input_tensor, &in_desc, &err_);

  // debug
  imgdnn_tensor_descriptor desc_1;
  imgdnnGetTensorDescriptor(reshaped_input, &desc_1);
  imgdnn_tensor_descriptor desc_2;
  imgdnnGetTensorDescriptor(weights_tensor, &desc_2);
  imgdnn_tensor_descriptor desc_3;
  imgdnnGetTensorDescriptor(bias_tensor, &desc_3);

  // handle weights [num_units, input_size] tensor
  /* const int order[] = { 1, 0 };
  auto isnu_weights_tensor = imgdnnNetworkTransposeOp(net,
                                                      weights_tensor,
                                                      order,
                                                      &err_);*/

  fcw_tensor = imgdnnNetworkBinaryOp(
      net_, reshaped_input, weights_tensor, IMGDNN_OPERATION_MATMUL, &err_);

  if (bias_tensor) {
    imgdnn_tensor fcw_int_tensor =
        imgdnnNetworkCastOp(net_, fcw_tensor, IMGDNN_TYPE_I32, nullptr, &err_);

    imgdnn_tensor_descriptor desc_4;
    imgdnnGetTensorDescriptor(fcw_int_tensor, &desc_4);

    fcb_tensor = imgdnnNetworkBinaryOp(
        net_, fcw_int_tensor, bias_tensor, IMGDNN_OPERATION_ADD, &err_);
  } else {
    fcb_tensor = fcw_tensor;
  }

  imgdnn_tensor_descriptor desc;
  imgdnnGetTensorDescriptor(input_tensor, &desc);
  if (desc.type == IMGDNN_TYPE_Q_I8 || desc.type == IMGDNN_TYPE_Q_U8) {
    fcb_tensor = imgdnnNetworkCastOp(
        net_, fcb_tensor, desc.type, &dst_quant_param, &err_);
  }

  return fcb_tensor;
}

imgdnn_tensor ImgdnnManager::createSoftmaxLayer(
    imgdnn_tensor input_tensor,
    float beta,
    unsigned int axis,
    imgdnn_quant_param dst_quant_param) {
  // debug
  imgdnn_tensor_descriptor desc_1;
  imgdnnGetTensorDescriptor(input_tensor, &desc_1);

  imgdnn_tensor softmax_tensor =
      imgdnnNetworkSoftmaxOp(net_, input_tensor, beta, axis, &err_);
  imgdnn_tensor_descriptor desc;
  imgdnnGetTensorDescriptor(input_tensor, &desc);
  if (desc.type == IMGDNN_TYPE_Q_I8 || desc.type == IMGDNN_TYPE_Q_U8) {
    softmax_tensor = imgdnnNetworkCastOp(
        net_, softmax_tensor, desc.type, &dst_quant_param, &err_);
  }

  imgdnn_tensor_descriptor desc_2;
  imgdnnGetTensorDescriptor(softmax_tensor, &desc_2);

  return softmax_tensor;
}

imgdnn_tensor ImgdnnManager::createScaleLayer(imgdnn_tensor input_tensor,
                                              bool with_biasscale,
                                              const void *const scale,
                                              const void *const bias) {
  imgdnn_tensor sc_tensor;
  imgdnn_tensor scale_tensor;
  imgdnn_tensor_descriptor sc_desc;

  imgdnn_tensor broadcast2_tensor;
  imgdnn_tensor broadcast3_tensor;

  unsigned int buffer_size;

  imgdnn_tensor_descriptor in_desc;
  imgdnnGetTensorDescriptor(input_tensor, &in_desc);

  sc_desc.dimensions = 2;
  sc_desc.type = in_desc.type;
  sc_desc.size[0] = in_desc.size[0];
  sc_desc.size[1] = in_desc.size[1];

  scale_tensor = createFixedInputTensor(&sc_desc, scale, true);

  broadcast2_tensor =
      imgdnnNetworkBroadcastOp(net_, scale_tensor, 2, in_desc.size[2], &err_);

  broadcast3_tensor = imgdnnNetworkBroadcastOp(
      net_, broadcast2_tensor, 3, in_desc.size[3], &err_);

  sc_tensor = imgdnnNetworkBinaryOp(
      net_, input_tensor, broadcast3_tensor, IMGDNN_OPERATION_MUL, &err_);

  if (with_biasscale) {
    imgdnn_tensor bsc_tensor;
    imgdnn_tensor biasscale_tensor;

    biasscale_tensor = createFixedInputTensor(&sc_desc, bias, true);

    broadcast2_tensor = imgdnnNetworkBroadcastOp(
        net_, biasscale_tensor, 2, in_desc.size[2], &err_);

    broadcast3_tensor = imgdnnNetworkBroadcastOp(
        net_, broadcast2_tensor, 3, in_desc.size[3], &err_);

    bsc_tensor = imgdnnNetworkBinaryOp(
        net_, sc_tensor, broadcast3_tensor, IMGDNN_OPERATION_ADD, &err_);
    return bsc_tensor;
  } else {
    return sc_tensor;
  }
}

imgdnn_network_object ImgdnnManager::createNetworkObject(
    unsigned int num_inputs,
    imgdnn_tensor *inputs,
    unsigned int num_outputs,
    imgdnn_tensor *outputs) {
  const imgdnn_network_object_flags flags = 0;

  std::string options_str;
  std::string ddk_root{"/home/jasonwang/imgtools/imagination_nna_sdk/"};
  // std::string ddk_root{STR2(IMAGINATION_NNA_SDK_ROOT)};
  std::string hwconfig =
      ddk_root + "nna-tools/config/mirage_hw_config06_23_2_6500_301.json";
  std::string mapconfig = ddk_root + "nna-tools/config/mapconfig_q8a.json";
  options_str += "-h " + hwconfig;
  options_str += " -m " + mapconfig;
  // options_str += " --dump_debug_binaries enabled";

  net_obj_ = imgdnnCreateNetworkObject(device_,
                                       context_,
                                       net_,
                                       num_inputs,
                                       inputs,
                                       num_outputs,
                                       outputs,
                                       flags,
                                       options_str.c_str(),
                                       &err_);
  ASSERT(err_ != IMGDNN_SUCCESS, "CreateNetworkObject failed!");
  return net_obj_;
}

}  // namespace imagination_nna
}  // namespace lite
}  // namespace paddle
