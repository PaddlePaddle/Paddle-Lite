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

#include "lite/backends/imagination_nna/imgdnn_manager.h"
#include <unistd.h>
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
      LOG(ERROR) << "unknown report flag in error callback";
  }

  LOG(ERROR) << msg_prefix << ": " << error_message;
}

ImgdnnManager::ImgdnnManager() {
  err_ = imgdnnSetErrorHandler(err_callback);
  net_ = imgdnnCreateNetwork(&err_);
  IMG_CHECK_SUCCESS(err_) << "CreateNetwork failed!";

  unsigned int num_devices;
  err_ = imgdnnGetDevices(
      IMGDNN_DEVICE_TYPE_ACCELERATOR, 1, &device_, &num_devices);
  IMG_CHECK_SUCCESS(err_) << "GetDevices failed!";
  context_ = imgdnnCreateContext(num_devices, &device_, 0, &err_);
  IMG_CHECK_SUCCESS(err_) << "CreateContext failed!";
  binding_ = imgdnnCreateBinding(&err_);
  IMG_CHECK_SUCCESS(err_) << "CreateBinding failed!";
}

imgdnn_tensor ImgdnnManager::ConvertQuantTensorType(
    imgdnn_tensor a_tensor, imgdnn_quant_param *dst_quant_param) {
  CHECK(dst_quant_param != NULL) << "dst_quant_param is NULL";

  imgdnn_tensor_descriptor desc;
  err_ = imgdnnGetTensorDescriptor(a_tensor, &desc);
  IMG_CHECK_SUCCESS(err_) << "imgdnn GetTensorDescriptor failed!";

  imgdnn_type dst_tensor_type;
  if (desc.type == IMGDNN_TYPE_Q_I8 || desc.type == IMGDNN_TYPE_Q_U8) {
    dst_tensor_type = desc.type;
  } else {
    dst_tensor_type = IMGDNN_TYPE_Q_U8;
  }

  imgdnn_tensor converted_tensor = imgdnnNetworkCastOp(
      net_, a_tensor, dst_tensor_type, dst_quant_param, &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn CastOp failed!";

  return converted_tensor;
}

bool ImgdnnManager::CheckConfigFileExists(const std::string &hwconfig,
                                          const std::string &mapconfig) {
  CHECK_EQ(access(hwconfig.c_str(), F_OK), 0)
      << "Could not find or access Imagination NNA hardware config file "
      << hwconfig;
  CHECK_EQ(access(mapconfig.c_str(), F_OK), 0)
      << "Could not find or access Imagination NNA mapping config file "
      << mapconfig;
  return true;
}

imgdnn_tensor ImgdnnManager::CreateConvolutionLayer(
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
  IMG_CHECK_SUCCESS(err_) << "imgdnn Convolution2dOp failed!";

  imgdnn_tensor conv2d_tensor = convw_tensor;
  if (bias_tensor) {
    imgdnn_tensor convw_int_tensor = imgdnnNetworkCastOp(
        net_, convw_tensor, IMGDNN_TYPE_I32, nullptr, &err_);
    IMG_CHECK_SUCCESS(err_) << "imgdnn CastOp failed!";

    imgdnn_tensor_descriptor bias_desc;
    imgdnnGetTensorDescriptor(convw_tensor, &bias_desc);
    IMG_CHECK_SUCCESS(err_) << "imgdnn GetTensorDescriptor failed!";

    imgdnn_tensor broadcast2_tensor;
    broadcast2_tensor = imgdnnNetworkBroadcastOp(
        net_, bias_tensor, 2, bias_desc.size[2], &err_);
    IMG_CHECK_SUCCESS(err_) << "imgdnn BroadcastOp failed!";

    imgdnn_tensor broadcast3_tensor;
    broadcast3_tensor = imgdnnNetworkBroadcastOp(
        net_, broadcast2_tensor, 3, bias_desc.size[3], &err_);
    IMG_CHECK_SUCCESS(err_) << "imgdnn BroadcastOp failed!";

    conv2d_tensor = imgdnnNetworkBinaryOp(
        net_, convw_int_tensor, broadcast3_tensor, IMGDNN_OPERATION_ADD, &err_);
    IMG_CHECK_SUCCESS(err_) << "imgdnn BinaryOp ADD failed!";
  }

  return ConvertQuantTensorType(conv2d_tensor, &dst_quant_param);
}

imgdnn_tensor ImgdnnManager::CreateBatchNormLayer(imgdnn_tensor input_tensor,
                                                  const void *const avg_in,
                                                  const void *const var_in,
                                                  const float eps) {
  imgdnn_tensor broadcast2_tensor;
  imgdnn_tensor broadcast3_tensor;

  imgdnn_tensor_descriptor in_desc;
  imgdnnGetTensorDescriptor(input_tensor, &in_desc);
  IMG_CHECK_SUCCESS(err_) << "imgdnn GetTensorDescriptor failed!";

  imgdnn_tensor_descriptor av_desc;
  av_desc.dimensions = 2;
  av_desc.type = in_desc.type;
  av_desc.size[0] = in_desc.size[0];
  av_desc.size[1] = in_desc.size[1];

  imgdnn_tensor average_tensor = CreateFixedInputTensor(&av_desc, avg_in, true);
  broadcast2_tensor =
      imgdnnNetworkBroadcastOp(net_, average_tensor, 2, in_desc.size[2], &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn BroadcastOp failed!";
  broadcast3_tensor = imgdnnNetworkBroadcastOp(
      net_, broadcast2_tensor, 3, in_desc.size[3], &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn BroadcastOp failed!";
  imgdnn_tensor bna_tensor = imgdnnNetworkBinaryOp(
      net_, input_tensor, broadcast3_tensor, IMGDNN_OPERATION_SUB, &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn BinaryOp SUB failed!";

  imgdnn_tensor_descriptor va_desc;
  va_desc.dimensions = 2;
  va_desc.type = in_desc.type;
  va_desc.size[0] = in_desc.size[0];
  va_desc.size[1] = in_desc.size[1];

  unsigned int buffer_size = imgdnnGetDescriptorSize(&va_desc, &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn GetDescriptorSize failed!";
  float *variance = reinterpret_cast<float *>(GetBufromPool(buffer_size));
  memcpy(variance, var_in, buffer_size);
  // Perform 1/sqrt(var+eps) and Update var_data.
  buffer_size /= sizeof(float);
  for (size_t i = 0; i < buffer_size; i++) {
    variance[i] = 1.0 / (sqrt(variance[i] + eps));
  }

  imgdnn_tensor variance_tensor =
      CreateFixedInputTensor(&va_desc, variance, false);
  broadcast2_tensor = imgdnnNetworkBroadcastOp(
      net_, variance_tensor, 2, in_desc.size[2], &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn BroadcastOp failed!";
  broadcast3_tensor = imgdnnNetworkBroadcastOp(
      net_, broadcast2_tensor, 3, in_desc.size[3], &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn BroadcastOp failed!";
  imgdnn_tensor bn_tensor = imgdnnNetworkBinaryOp(
      net_, bna_tensor, broadcast3_tensor, IMGDNN_OPERATION_MUL, &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn BinaryOp MUL failed!";

  return bn_tensor;
}

imgdnn_tensor ImgdnnManager::CreatePoolingLayer(
    imgdnn_tensor in_tensor,
    imgdnn_quant_param dst_quant_param,
    const unsigned int size[2],
    const unsigned int stride[2],
    const unsigned int pad_to_begin[2],
    const unsigned int pad_to_end[2],
    imgdnn_pooling_type type) {
  imgdnn_tensor pool_tensor = imgdnnNetworkPooling2dOp_v2(
      net_, in_tensor, size, stride, pad_to_begin, pad_to_end, type, &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn Pooling2dOp failed!";

  return ConvertQuantTensorType(pool_tensor, &dst_quant_param);
}

imgdnn_tensor ImgdnnManager::CreateFullyConnectedLayer(
    imgdnn_tensor input_tensor,
    imgdnn_tensor weights_tensor,
    imgdnn_tensor bias_tensor,
    imgdnn_quant_param dst_quant_param) {
  imgdnn_tensor_descriptor in_desc;
  imgdnnGetTensorDescriptor(input_tensor, &in_desc);
  IMG_CHECK_SUCCESS(err_) << "imgdnn GetTensorDescriptor failed!";

  // int flatten_dim = 1
  for (unsigned i = 2; i < in_desc.dimensions; ++i)
    in_desc.size[1] *= in_desc.size[i];
  in_desc.dimensions = 2;

  auto reshaped_input =
      imgdnnNetworkReshapeOp(net_, input_tensor, &in_desc, &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn ReshapeOp failed!";

  imgdnn_tensor fcw_tensor = imgdnnNetworkBinaryOp(
      net_, reshaped_input, weights_tensor, IMGDNN_OPERATION_MATMUL, &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn BinaryOp MATMUL failed!";

  imgdnn_tensor fcb_tensor;
  if (bias_tensor) {
    imgdnn_tensor fcw_int_tensor =
        imgdnnNetworkCastOp(net_, fcw_tensor, IMGDNN_TYPE_I32, nullptr, &err_);
    IMG_CHECK_SUCCESS(err_) << "imgdnn CastOp failed!";
    fcb_tensor = imgdnnNetworkBinaryOp(
        net_, fcw_int_tensor, bias_tensor, IMGDNN_OPERATION_ADD, &err_);
    IMG_CHECK_SUCCESS(err_) << "imgdnn BinaryOp ADD failed!";
  } else {
    fcb_tensor = fcw_tensor;
  }

  return ConvertQuantTensorType(fcb_tensor, &dst_quant_param);
}

imgdnn_tensor ImgdnnManager::CreateSoftmaxLayer(
    imgdnn_tensor input_tensor,
    float beta,
    unsigned int axis,
    imgdnn_quant_param dst_quant_param) {
  imgdnn_tensor softmax_tensor =
      imgdnnNetworkSoftmaxOp(net_, input_tensor, beta, axis, &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn SoftmaxOp failed!";
  return ConvertQuantTensorType(softmax_tensor, &dst_quant_param);
}

imgdnn_tensor ImgdnnManager::CreateScaleLayer(imgdnn_tensor input_tensor,
                                              bool with_biasscale,
                                              const void *const scale,
                                              const void *const bias) {
  imgdnn_tensor broadcast2_tensor;
  imgdnn_tensor broadcast3_tensor;

  imgdnn_tensor_descriptor in_desc;
  imgdnnGetTensorDescriptor(input_tensor, &in_desc);
  IMG_CHECK_SUCCESS(err_) << "imgdnn GetTensorDescriptor failed!";

  imgdnn_tensor_descriptor sc_desc;
  sc_desc.dimensions = 2;
  sc_desc.type = in_desc.type;
  sc_desc.size[0] = in_desc.size[0];
  sc_desc.size[1] = in_desc.size[1];

  imgdnn_tensor scale_tensor = CreateFixedInputTensor(&sc_desc, scale, true);
  broadcast2_tensor =
      imgdnnNetworkBroadcastOp(net_, scale_tensor, 2, in_desc.size[2], &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn BroadcastOp failed!";
  broadcast3_tensor = imgdnnNetworkBroadcastOp(
      net_, broadcast2_tensor, 3, in_desc.size[3], &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn BroadcastOp failed!";
  imgdnn_tensor sc_tensor = imgdnnNetworkBinaryOp(
      net_, input_tensor, broadcast3_tensor, IMGDNN_OPERATION_MUL, &err_);
  IMG_CHECK_SUCCESS(err_) << "imgdnn BinaryOp MUL failed!";

  if (with_biasscale) {
    imgdnn_tensor biasscale_tensor =
        CreateFixedInputTensor(&sc_desc, bias, true);
    broadcast2_tensor = imgdnnNetworkBroadcastOp(
        net_, biasscale_tensor, 2, in_desc.size[2], &err_);
    IMG_CHECK_SUCCESS(err_) << "imgdnn BroadcastOp failed!";
    broadcast3_tensor = imgdnnNetworkBroadcastOp(
        net_, broadcast2_tensor, 3, in_desc.size[3], &err_);
    IMG_CHECK_SUCCESS(err_) << "imgdnn BroadcastOp failed!";
    sc_tensor = imgdnnNetworkBinaryOp(
        net_, sc_tensor, broadcast3_tensor, IMGDNN_OPERATION_ADD, &err_);
    IMG_CHECK_SUCCESS(err_) << "imgdnn BinaryOp ADD failed!";
  }

  return sc_tensor;
}

imgdnn_network_object ImgdnnManager::CreateNetworkObject(
    unsigned int num_inputs,
    imgdnn_tensor *inputs,
    unsigned int num_outputs,
    imgdnn_tensor *outputs) {
  const imgdnn_network_object_flags flags = 0;

  const std::string hwconfig =
      "nna_config/mirage_hw_config06_23_2_6500_301.json";
  const std::string mapconfig = "nna_config/mapconfig_q8a.json";

  CheckConfigFileExists(hwconfig, mapconfig);

  std::string options_str;
  options_str += "-h " + hwconfig;
  options_str += " -m " + mapconfig;
  //  Add " --dump_debug_binaries enabled" to options_str if need debug info.

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
  IMG_CHECK_SUCCESS(err_) << "CreateNetworkObject failed!";
  return net_obj_;
}

}  // namespace imagination_nna
}  // namespace lite
}  // namespace paddle
