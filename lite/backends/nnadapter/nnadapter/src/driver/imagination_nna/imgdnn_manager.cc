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

#include "driver/imagination_nna/imgdnn_manager.h"
#include <dlfcn.h>
#include <unistd.h>
#include <utility>

namespace nnadapter {
namespace imagination_nna {

static void error_callback(imgdnn_report_flags flags,
                           const char **tensor_names,
                           int num_tensor_names,
                           imgdnn_err_code error_code,
                           const char *error_message) {
  std::string msg_prefix;
  switch (flags) {
    case imgdnn_report_flags::IMGDNN_REPORT_ERROR:
      NNADAPTER_LOG(ERROR) << error_message;
      break;
    case imgdnn_report_flags::IMGDNN_REPORT_VERBOSE:
      NNADAPTER_LOG(INFO) << error_message;
      break;
    case imgdnn_report_flags::IMGDNN_REPORT_INFO:
      NNADAPTER_LOG(INFO) << error_message;
      break;
    case imgdnn_report_flags::IMGDNN_REPORT_WARNING:
      NNADAPTER_LOG(WARNING) << error_message;
      break;
    default:
      NNADAPTER_LOG(FATAL) << "Unknown report flag " << static_cast<int>(flags)
                           << " in error callback!";
  }
}

ImgdnnManager::ImgdnnManager() {
  imgdnn_err_code error_code;
  IMGDNN_CHECK_SUCCESS(imgdnnSetErrorHandler(error_callback));
  network_ = imgdnnCreateNetwork(&error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnCreateNetwork!";
  unsigned int num_devices;
  IMGDNN_CHECK_SUCCESS(imgdnnGetDevices(
      IMGDNN_DEVICE_TYPE_ACCELERATOR, 1, &device_, &num_devices));
  context_ = imgdnnCreateContext(num_devices, &device_, 0, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnCreateContext!";
  binding_ = imgdnnCreateBinding(&error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnCreateBinding!";
}

ImgdnnManager::~ImgdnnManager() {
  if (network_object_) {
    IMGDNN_CHECK_SUCCESS(imgdnnNetworkObjectDestroy(network_object_));
  }
  if (context_) {
    IMGDNN_CHECK_SUCCESS(imgdnnContextDestroy(context_));
  }
  if (binding_) {
    IMGDNN_CHECK_SUCCESS(imgdnnBindingDestroy(binding_));
  }
  if (network_) {
    IMGDNN_CHECK_SUCCESS(imgdnnNetworkDestroy(network_));
  }
}

uint8_t *ImgdnnManager::GetBuffer(size_t size) {
  buffers_.emplace_back(new uint8_t[size]);
  return buffers_.back().get();
}

imgdnn_tensor ImgdnnManager::CreateInputTensor(imgdnn_tensor_descriptor *desc) {
  imgdnn_err_code error_code;
  auto input_tensor = imgdnnNetworkInput(network_, desc, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnCreateContext!";
  return input_tensor;
}

imgdnn_tensor ImgdnnManager::CreateFixedInputTensor(
    imgdnn_tensor_descriptor *desc, const void *const data, bool copy) {
  imgdnn_err_code error_code;
  imgdnn_tensor fixed_input_tensor;
  if (copy) {
    auto size = imgdnnGetDescriptorSize(desc, &error_code);
    IMGDNN_CHECK_SUCCESS(error_code)
        << "Failed to call imgdnnGetDescriptorSize!";
    auto buffer = GetBuffer(size);
    memcpy(buffer, data, size);
    fixed_input_tensor =
        imgdnnNetworkFixedInput(network_, desc, buffer, &error_code);
  } else {
    fixed_input_tensor =
        imgdnnNetworkFixedInput(network_, desc, data, &error_code);
  }
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkFixedInput!";
  return fixed_input_tensor;
}

imgdnn_tensor ImgdnnManager::CreateConvolutionLayer(
    imgdnn_tensor input_tensor,
    imgdnn_tensor weights_tensor,
    imgdnn_tensor bias_tensor,
    imgdnn_quant_param quant_param,
    unsigned int stride[2],
    unsigned int pad_begin[2],
    unsigned int pad_end[2],
    unsigned int dilation[2],
    bool use_dwconv) {
  imgdnn_err_code error_code;
  imgdnn_tensor output_tensor;
  if (use_dwconv) {
    // Transpose weight tensor
    int order[4] = {1, 0, 2, 3};
    weights_tensor =
        imgdnnNetworkTransposeOp(network_, weights_tensor, order, &error_code);
    IMGDNN_CHECK_SUCCESS(error_code)
        << "Failed to call imgdnnNetworkTransposeOp!";
    output_tensor = imgdnnNetworkDepthConvolution2dOp_v2(network_,
                                                         input_tensor,
                                                         weights_tensor,
                                                         stride,
                                                         pad_begin,
                                                         pad_end,
                                                         dilation,
                                                         &error_code);
    IMGDNN_CHECK_SUCCESS(error_code)
        << "Failed to call imgdnnNetworkDepthConvolution2dOp_v2!";
  } else {
    output_tensor = imgdnnNetworkConvolution2dOp_v2(network_,
                                                    input_tensor,
                                                    weights_tensor,
                                                    stride,
                                                    pad_begin,
                                                    pad_end,
                                                    dilation,
                                                    &error_code);
    IMGDNN_CHECK_SUCCESS(error_code)
        << "Failed to call imgdnnNetworkConvolution2dOp_v2!";
  }
  if (bias_tensor) {
    // Broadcast bias tensor on height and width dimension
    imgdnn_tensor_descriptor output_desc;
    IMGDNN_CHECK_SUCCESS(
        imgdnnGetTensorDescriptor(output_tensor, &output_desc));
    bias_tensor = imgdnnNetworkBroadcastOp(
        network_, bias_tensor, 2, output_desc.size[2], &error_code);
    IMGDNN_CHECK_SUCCESS(error_code)
        << "Failed to call imgdnnNetworkBroadcastOp!";
    bias_tensor = imgdnnNetworkBroadcastOp(
        network_, bias_tensor, 3, output_desc.size[3], &error_code);
    IMGDNN_CHECK_SUCCESS(error_code)
        << "Failed to call imgdnnNetworkBroadcastOp!";
    output_tensor = imgdnnNetworkCastOp(
        network_, output_tensor, IMGDNN_TYPE_I32, nullptr, &error_code);
    IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkCastOp!";
    // Add bias tensor
    output_tensor = imgdnnNetworkBinaryOp(network_,
                                          output_tensor,
                                          bias_tensor,
                                          IMGDNN_OPERATION_ADD,
                                          &error_code);
    IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkBinaryOp!";
  }
  return ConvertQuantTensorType(output_tensor, &quant_param);
}

imgdnn_tensor ImgdnnManager::CreateBatchNormLayer(
    imgdnn_tensor input_tensor,
    const void *const mean_data,
    const void *const variance_data,
    const float eps) {
  imgdnn_err_code error_code;
  imgdnn_tensor_descriptor input_desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(input_tensor, &input_desc));
  // Mean tensor
  imgdnn_tensor_descriptor mean_desc;
  mean_desc.dimensions = 2;
  mean_desc.type = input_desc.type;
  mean_desc.size[0] = input_desc.size[0];
  mean_desc.size[1] = input_desc.size[1];
  auto mean_tensor = CreateFixedInputTensor(&mean_desc, mean_data, true);
  mean_tensor = imgdnnNetworkBroadcastOp(
      network_, mean_tensor, 2, input_desc.size[2], &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnNetworkBroadcastOp!";
  mean_tensor = imgdnnNetworkBroadcastOp(
      network_, mean_tensor, 3, input_desc.size[3], &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnNetworkBroadcastOp!";
  auto output_tensor = imgdnnNetworkBinaryOp(
      network_, input_tensor, mean_tensor, IMGDNN_OPERATION_SUB, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnNetworkBroadcastOp!";
  // Variance tensor
  imgdnn_tensor_descriptor variance_desc;
  variance_desc.dimensions = 2;
  variance_desc.type = input_desc.type;
  variance_desc.size[0] = input_desc.size[0];
  variance_desc.size[1] = input_desc.size[1];
  size_t variance_size = imgdnnGetDescriptorSize(&variance_desc, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnGetDescriptorSize!";
  auto variance_buffer = reinterpret_cast<float *>(GetBuffer(variance_size));
  memcpy(variance_buffer, variance_data, variance_size);
  // Perform 1/sqrt(variance + eps) and update variance_data.
  variance_size /= sizeof(float);
  for (size_t i = 0; i < variance_size; i++) {
    variance_buffer[i] = 1.0 / (sqrt(variance_buffer[i] + eps));
  }
  auto variance_tensor =
      CreateFixedInputTensor(&variance_desc, variance_buffer, false);
  variance_tensor = imgdnnNetworkBroadcastOp(
      network_, variance_tensor, 2, input_desc.size[2], &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnNetworkBroadcastOp!";
  variance_tensor = imgdnnNetworkBroadcastOp(
      network_, variance_tensor, 3, input_desc.size[3], &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnNetworkBroadcastOp!";
  output_tensor = imgdnnNetworkBinaryOp(network_,
                                        output_tensor,
                                        variance_tensor,
                                        IMGDNN_OPERATION_MUL,
                                        &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnNetworkBroadcastOp!";
  return output_tensor;
}

imgdnn_tensor ImgdnnManager::CreatePoolingLayer(
    imgdnn_tensor input_tensor,
    imgdnn_quant_param quant_param,
    const unsigned int size[2],
    const unsigned int stride[2],
    const unsigned int pad_to_begin[2],
    const unsigned int pad_to_end[2],
    bool count_include_pad,
    imgdnn_pooling_type type) {
  imgdnn_err_code error_code;
  auto output_tensor = imgdnnNetworkPooling2dOp_v3(network_,
                                                   input_tensor,
                                                   size,
                                                   stride,
                                                   pad_to_begin,
                                                   pad_to_end,
                                                   type,
                                                   count_include_pad,
                                                   &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnNetworkPooling2dOp_v3!";
  return output_tensor;
}

imgdnn_tensor ImgdnnManager::CreateFullyConnectedLayer(
    imgdnn_tensor input_tensor,
    imgdnn_tensor weights_tensor,
    imgdnn_tensor bias_tensor,
    imgdnn_quant_param quant_param) {
  imgdnn_err_code error_code;
  imgdnn_tensor_descriptor input_desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(input_tensor, &input_desc));
  // Flatten the input tensor from dimension 2 (Only supports flatten_dim=1)
  for (unsigned i = 2; i < input_desc.dimensions; i++) {
    input_desc.size[1] *= input_desc.size[i];
  }
  input_desc.dimensions = 2;
  input_tensor =
      imgdnnNetworkReshapeOp(network_, input_tensor, &input_desc, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkReshapeOp!";
  // Implement the FullyConnected layer using MatMul
  auto output_tensor = imgdnnNetworkBinaryOp(network_,
                                             input_tensor,
                                             weights_tensor,
                                             IMGDNN_OPERATION_MATMUL,
                                             &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkBinaryOp!";
  // Add bias tensor
  if (bias_tensor) {
    output_tensor = imgdnnNetworkCastOp(
        network_, output_tensor, IMGDNN_TYPE_I32, nullptr, &error_code);
    IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkCastOp!";
    output_tensor = imgdnnNetworkBinaryOp(network_,
                                          output_tensor,
                                          bias_tensor,
                                          IMGDNN_OPERATION_ADD,
                                          &error_code);
    IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkBinaryOp!";
  }
  return ConvertQuantTensorType(output_tensor, &quant_param);
}

imgdnn_tensor ImgdnnManager::CreateMatMulLayer(imgdnn_tensor x_tensor,
                                               imgdnn_tensor y_tensor,
                                               imgdnn_quant_param quant_param) {
  imgdnn_err_code error_code;
  imgdnn_tensor_descriptor x_desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(x_tensor, &x_desc));
  imgdnn_tensor_descriptor y_desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(y_tensor, &y_desc));
  NNADAPTER_CHECK_EQ(x_desc.dimensions, 2)
      << "Imgination NNA does not support the dimension of x != 2";
  NNADAPTER_CHECK_EQ(y_desc.dimensions, 2)
      << "Imgination NNA does not support the dimension of y != 2";
  x_tensor = imgdnnNetworkReshapeOp(network_, x_tensor, &x_desc, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkReshapeOp!";
  auto output_tensor = imgdnnNetworkBinaryOp(
      network_, x_tensor, y_tensor, IMGDNN_OPERATION_MATMUL, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkBinaryOp!";
  return ConvertQuantTensorType(output_tensor, &quant_param);
}

imgdnn_tensor ImgdnnManager::CreateElementwiseOpsLayer(
    imgdnn_tensor input0_tensor,
    imgdnn_tensor input1_tensor,
    imgdnn_operation_binary imgdnn_operation,
    imgdnn_quant_param quant_param) {
  imgdnn_err_code error_code;
  imgdnn_tensor_descriptor input0_desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(input0_tensor, &input0_desc));
  imgdnn_tensor_descriptor input1_desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(input1_tensor, &input1_desc));
  auto input0_type = input0_desc.type;
  auto input1_type = input1_desc.type;
  if (input0_type != input1_type) {
    if (input0_type != IMGDNN_TYPE_I32) {
      input0_tensor = imgdnnNetworkCastOp(
          network_, input0_tensor, IMGDNN_TYPE_I32, nullptr, &error_code);
      IMGDNN_CHECK_SUCCESS(error_code)
          << "Failed to call imgdnnNetworkCastOp for input0_tensor!";
    }
    if (input1_type != IMGDNN_TYPE_I32) {
      input1_tensor = imgdnnNetworkCastOp(
          network_, input1_tensor, IMGDNN_TYPE_I32, nullptr, &error_code);
      IMGDNN_CHECK_SUCCESS(error_code)
          << "Failed to call imgdnnNetworkCastOp for input1_tensor!";
    }
  }
  auto output_tensor = imgdnnNetworkBinaryOp(
      network_, input0_tensor, input1_tensor, imgdnn_operation, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkBinaryOp!";
  return ConvertQuantTensorType(output_tensor, &quant_param);
}

imgdnn_tensor ImgdnnManager::CreateReshapeLayer(
    imgdnn_tensor input_tensor,
    unsigned int *shape,
    uint32_t shape_count,
    imgdnn_quant_param quant_param) {
  imgdnn_err_code error_code;
  imgdnn_tensor_descriptor shape_desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(input_tensor, &shape_desc));
  shape_desc.dimensions = shape_count;
  for (unsigned i = 0; i < shape_count; i++) {
    shape_desc.size[i] = shape[i];
  }
  auto output_tensor =
      imgdnnNetworkReshapeOp(network_, input_tensor, &shape_desc, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkReshapeOp!";
  return ConvertQuantTensorType(output_tensor, &quant_param);
}

imgdnn_tensor ImgdnnManager::CreateSoftmaxLayer(
    imgdnn_tensor input_tensor,
    float beta,
    unsigned int axis,
    imgdnn_quant_param quant_param) {
  imgdnn_err_code error_code;
  auto output_tensor =
      imgdnnNetworkSoftmaxOp(network_, input_tensor, beta, axis, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkSoftmaxOp!";
  return ConvertQuantTensorType(output_tensor, &quant_param);
}

imgdnn_tensor ImgdnnManager::CreateScaleLayer(imgdnn_tensor input_tensor,
                                              bool with_bias,
                                              const void *const scale_data,
                                              const void *const bias_data) {
  imgdnn_err_code error_code;
  imgdnn_tensor_descriptor input_desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(input_tensor, &input_desc));
  // Scale tensor
  imgdnn_tensor_descriptor scale_desc;
  scale_desc.dimensions = 2;
  scale_desc.type = input_desc.type;
  scale_desc.size[0] = input_desc.size[0];
  scale_desc.size[1] = input_desc.size[1];
  auto scale_tensor = CreateFixedInputTensor(&scale_desc, scale_data, true);
  // Broadcast scale tensor on height and width dimension
  scale_tensor = imgdnnNetworkBroadcastOp(
      network_, scale_tensor, 2, input_desc.size[2], &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnNetworkBroadcastOp!";
  scale_tensor = imgdnnNetworkBroadcastOp(
      network_, scale_tensor, 3, input_desc.size[3], &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnNetworkBroadcastOp!";
  // Implement Scale layer using MUL
  auto output_tensor = imgdnnNetworkBinaryOp(
      network_, input_tensor, scale_tensor, IMGDNN_OPERATION_MUL, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkBinaryOp!";
  // Add bias
  if (with_bias) {
    auto bias_tensor = CreateFixedInputTensor(&scale_desc, bias_data, true);
    bias_tensor = imgdnnNetworkBroadcastOp(
        network_, bias_tensor, 2, input_desc.size[2], &error_code);
    IMGDNN_CHECK_SUCCESS(error_code)
        << "Failed to call imgdnnNetworkBroadcastOp!";
    bias_tensor = imgdnnNetworkBroadcastOp(
        network_, bias_tensor, 3, input_desc.size[3], &error_code);
    IMGDNN_CHECK_SUCCESS(error_code)
        << "Failed to call imgdnnNetworkBroadcastOp!";
    output_tensor = imgdnnNetworkBinaryOp(network_,
                                          output_tensor,
                                          bias_tensor,
                                          IMGDNN_OPERATION_ADD,
                                          &error_code);
    IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkBinaryOp!";
  }
  return output_tensor;
}

imgdnn_tensor ImgdnnManager::CreateReLULayer(imgdnn_tensor input_tensor,
                                             bool has_min_clamp,
                                             float min_clamp,
                                             bool has_max_clamp,
                                             float max_clamp,
                                             float negative_slope) {
  imgdnn_err_code error_code;
  auto output_tensor = imgdnnNetworkReLUOp(network_,
                                           input_tensor,
                                           has_min_clamp,
                                           min_clamp,
                                           has_max_clamp,
                                           max_clamp,
                                           negative_slope,
                                           &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkReLUOp!";
  imgdnn_tensor_descriptor input_desc, output_desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(input_tensor, &input_desc));
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(output_tensor, &output_desc));
  if (output_desc.type != input_desc.type) {
    output_tensor = imgdnnNetworkCastOp(network_,
                                        output_tensor,
                                        input_desc.type,
                                        &input_desc.quant_param,
                                        &error_code);
    IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkCastOp!";
  }
  return output_tensor;
}

imgdnn_tensor ImgdnnManager::ConvertQuantTensorType(
    imgdnn_tensor tensor, imgdnn_quant_param *quant_param) {
  imgdnn_err_code error_code;
  NNADAPTER_CHECK(quant_param);
  imgdnn_tensor_descriptor desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(tensor, &desc));
  imgdnn_type type;
  if (desc.type == IMGDNN_TYPE_Q_I8 || desc.type == IMGDNN_TYPE_Q_U8) {
    type = desc.type;
  } else {
    type = IMGDNN_TYPE_Q_U8;
  }
  auto quant_tensor =
      imgdnnNetworkCastOp(network_, tensor, type, quant_param, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnNetworkCastOp!";
  return quant_tensor;
}

bool ImgdnnManager::CheckConfigFileExists(const std::string &hw_config_path,
                                          const std::string &map_config_path) {
  NNADAPTER_CHECK_EQ(access(hw_config_path.c_str(), F_OK), 0)
      << "Could not find or access Imagination NNA hardware config file "
      << hw_config_path;
  NNADAPTER_CHECK_EQ(access(map_config_path.c_str(), F_OK), 0)
      << "Could not find or access Imagination NNA mapping config file "
      << map_config_path;
  return true;
}

imgdnn_memory ImgdnnManager::ImportMemory(
    void *buffer, size_t size, imgdnn_import_mem_type import_mem_type) {
  imgdnn_err_code error_code;
  auto memory =
      imgdnnImportMemory(context_, buffer, size, import_mem_type, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnImportMemory!";
  return memory;
}

imgdnn_memory ImgdnnManager::AllocateMemory(size_t size) {
  imgdnn_err_code error_code;
  auto memory = imgdnnAllocateMemory(context_, size, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnImportMemory!";
  return memory;
}

void ImgdnnManager::DestroyMemory(imgdnn_memory memory) {
  IMGDNN_CHECK_SUCCESS(imgdnnMemoryDestroy(memory));
}

void *ImgdnnManager::LockMemory(imgdnn_memory memory,
                                imgdnn_lock_access lock_access) {
  imgdnn_err_code error_code;
  void *buffer = imgdnnMemoryLock(memory, lock_access, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnMemoryLock!";
  return buffer;
}

void ImgdnnManager::UnlockMemory(imgdnn_memory memory) {
  IMGDNN_CHECK_SUCCESS(imgdnnMemoryUnlock(memory));
}

imgdnn_network_object ImgdnnManager::CreateNetworkObject(
    unsigned int num_inputs,
    imgdnn_tensor *inputs,
    unsigned int num_outputs,
    imgdnn_tensor *outputs) {
  imgdnn_err_code error_code;
  const imgdnn_network_object_flags flags = 0;
  // Get the directory of current module
  std::string cur_dir = ".";
  Dl_info dl_info;
  dladdr(reinterpret_cast<void *>(error_callback), &dl_info);
  if (dl_info.dli_fname) {
    std::string dli_fname = dl_info.dli_fname;
    const size_t last_slash_idx = dli_fname.rfind('/');
    if (std::string::npos != last_slash_idx) {
      cur_dir = dli_fname.substr(0, last_slash_idx);
    }
  }
  const std::string hw_config_path =
      cur_dir + "/nna_config/mirage_hw_config06_23_2_6500_301.json";
  const std::string map_config_path =
      cur_dir + "/nna_config/mapconfig_q8a.json";
  CheckConfigFileExists(hw_config_path, map_config_path);
  std::string options;
  options += "-h " + hw_config_path;
  options += " -m " + map_config_path;
  // Add " --dump_debug_binaries enabled" to options if need debug info.
  network_object_ = imgdnnCreateNetworkObject(device_,
                                              context_,
                                              network_,
                                              num_inputs,
                                              inputs,
                                              num_outputs,
                                              outputs,
                                              flags,
                                              options.c_str(),
                                              &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnCreateNetworkObject!";
  return network_object_;
}

void ImgdnnManager::ExecuteNetworkObject(bool blocking_execute,
                                         unsigned int num_events_in_wait_list,
                                         const imgdnn_event event_wait_list[],
                                         imgdnn_event *event) {
  IMGDNN_CHECK_SUCCESS(imgdnnNetworkObjectExecute(network_object_,
                                                  binding_,
                                                  blocking_execute,
                                                  num_events_in_wait_list,
                                                  event_wait_list,
                                                  event));
}

void ImgdnnManager::GetNetworkObjectInputs(unsigned int max_inputs,
                                           imgdnn_input inputs[],
                                           unsigned int *num_inputs) {
  NNADAPTER_CHECK(network_object_);
  IMGDNN_CHECK_SUCCESS(imgdnnNetworkObjectGetInputs(
      network_object_, max_inputs, inputs, num_inputs));
}

void ImgdnnManager::GetNetworkObjectOutputs(unsigned int max_outputs,
                                            imgdnn_output outputs[],
                                            unsigned int *num_outputs) {
  NNADAPTER_CHECK(network_object_);
  IMGDNN_CHECK_SUCCESS(imgdnnNetworkObjectGetOutputs(
      network_object_, max_outputs, outputs, num_outputs));
}

imgdnn_tensor_descriptor ImgdnnManager::GetInputDescriptor(imgdnn_input input) {
  imgdnn_err_code error_code;
  imgdnn_tensor_descriptor input_desc =
      imgdnnGetInputDescriptor(input, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnGetInputDescriptor!";
  return input_desc;
}

imgdnn_tensor_descriptor ImgdnnManager::GetOutputDescriptor(
    imgdnn_output output) {
  imgdnn_err_code error_code;
  imgdnn_tensor_descriptor output_desc =
      imgdnnGetOutputDescriptor(output, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code)
      << "Failed to call imgdnnGetOutputDescriptor!";
  return output_desc;
}

imgdnn_tensor_descriptor ImgdnnManager::GetTensorDescriptor(
    imgdnn_tensor tensor) {
  imgdnn_tensor_descriptor desc;
  IMGDNN_CHECK_SUCCESS(imgdnnGetTensorDescriptor(tensor, &desc));
  return desc;
}

size_t ImgdnnManager::GetDescriptorSize(
    const imgdnn_tensor_descriptor *const descriptor) {
  imgdnn_err_code error_code;
  auto size = imgdnnGetDescriptorSize(descriptor, &error_code);
  IMGDNN_CHECK_SUCCESS(error_code) << "Failed to call imgdnnGetDescriptorSize!";
  return size;
}

void ImgdnnManager::AddBindingInput(imgdnn_input input, imgdnn_memory memory) {
  IMGDNN_CHECK_SUCCESS(imgdnnBindingAddInput(binding_, input, memory));
}

void ImgdnnManager::AddBindingOutput(imgdnn_output output,
                                     imgdnn_memory memory) {
  IMGDNN_CHECK_SUCCESS(imgdnnBindingAddOutput(binding_, output, memory));
}

}  // namespace imagination_nna
}  // namespace nnadapter
