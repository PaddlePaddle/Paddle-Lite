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

#include "lite/api/paddle_place.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void copy_properties(operators::IoCopyParam& param) {  // NOLINT
  param.y->set_persistable(param.x->persistable());
  auto out_lod = param.y->mutable_lod();
  *out_lod = param.x->lod();
  param.y->ZynqTensor()->copyScaleFrom(param.x->ZynqTensor());
}

/*
 * This kernel copies a tensor from host to FPGA space.
 */
class IoCopyHostCHWToFpgaHWCCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kFPGA));
    param.x->ZynqTensor()->flush();

    if (param.x->ZynqTensor()->dataType() == zynqmp::INT32) {
      param.y->mutable_data<int>();
      param.y->ZynqTensor()->copyFrom(param.x->ZynqTensor());
      param.y->ZynqTensor()->flush();
      copy_properties(param);
      return;
    }

    param.y->mutable_data<float16>();
    param.y->ZynqTensor()->setDataLocation(zynqmp::Device);
    if (param.x->ZynqTensor()->aligned() &&
        param.x->ZynqTensor()->shape().shouldAlign()) {
      zynqmp::Tensor tempTensor;
      tempTensor.mutableData<float16>(zynqmp::FP16,
                                      param.x->ZynqTensor()->shape());
      tempTensor.copyFrom(param.x->ZynqTensor());
      tempTensor.setAligned(true);
      tempTensor.unalignImage();
      tempTensor.flush();
      param.y->ZynqTensor()->copyFrom(&tempTensor);
    } else {
      param.y->ZynqTensor()->copyFrom(param.x->ZynqTensor());
    }
    copy_properties(param);
    param.y->ZynqTensor()->invalidate();
  }

  std::string doc() const override { return "Copy IO from HOST to FPGA"; }
};

/*
 * This kernel copies a tensor from FPGA to host space.
 */
class IoCopyFpgaToHostCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kFPGA));

    param.x->ZynqTensor()->syncToDevice();
    param.y->mutable_data<float>();
    param.y->ZynqTensor()->setDataType(zynqmp::FP32);
    param.y->ZynqTensor()->setDataLocation(zynqmp::CPU);

    if (param.x->ZynqTensor()->aligned() &&
        param.x->ZynqTensor()->shape().shouldAlign()) {
      zynqmp::Tensor tempTensor;
      tempTensor.mutableData<float16>(zynqmp::FP16,
                                      param.x->ZynqTensor()->shape());
      tempTensor.copyFrom(param.x->ZynqTensor());
      tempTensor.setAligned(true);
      tempTensor.unalignImage();
      param.y->ZynqTensor()->copyFrom(&tempTensor);
    } else {
      param.y->ZynqTensor()->copyFrom(param.x->ZynqTensor());
    }

    param.y->ZynqTensor()->invalidate();
    copy_properties(param);
  }
  std::string doc() const override { return "Copy IO from FPGA to HOST"; }
};

void hwc_to_chw(float* chw_data,
                float* hwc_data,
                int num,
                int channel,
                int height,
                int width) {
  int chw = channel * height * width;
  int wc = width * channel;
  int wh = width * height;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channel; c++) {
          chw_data[n * chw + c * wh + h * width + w] = hwc_data[index];
          index++;
        }
      }
    }
  }
}

class IoCopyFpgaToHostCHWCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kFPGA));

    param.x->ZynqTensor()->syncToDevice();
    if (param.x->ZynqTensor()->dataType() == zynqmp::INT32) {
      param.y->mutable_data<int32_t>();
      param.y->ZynqTensor()->copyFrom(param.x->ZynqTensor());
      return;
    }

    Tensor hwc;
    hwc.Resize(param.y->dims());
    float* hwc_data = hwc.mutable_data<float>();
    float* chw_data = param.y->mutable_data<float>();
    param.y->ZynqTensor()->setDataType(zynqmp::FP32);

    hwc.ZynqTensor()->setDataLocation(zynqmp::CPU);
    param.y->ZynqTensor()->setDataLocation(zynqmp::CPU);

    if (param.x->ZynqTensor()->aligned() &&
        param.x->ZynqTensor()->shape().shouldAlign()) {
      zynqmp::Tensor tempTensor;
      tempTensor.mutableData<float16>(zynqmp::FP16,
                                      param.x->ZynqTensor()->shape());
      tempTensor.copyFrom(param.x->ZynqTensor());
      tempTensor.setAligned(true);
      tempTensor.unalignImage();

      hwc.ZynqTensor()->copyFrom(&tempTensor);
    } else {
      float16* in_data = param.x->ZynqTensor()->data<float16>();
      param.x->ZynqTensor()->flush();
      float max = 0;

      for (int i = 0; i < param.x->dims().production(); i++) {
        float value = zynqmp::half_to_float(in_data[i]);
        hwc_data[i] = value;
        if (value < 0) {
          value = -value;
        }
        if (value > max) {
          max = value;
        }
      }
      param.x->ZynqTensor()->scale()[0] = max / 127;
      param.x->ZynqTensor()->scale()[1] = 127 / max;
    }

    int num = 1;
    int channel = 1;
    int height = 1;
    int width = 1;

    auto dims = param.y->ZynqTensor()->shape();

    hwc_to_chw(chw_data,
               hwc_data,
               dims.num(),
               dims.channel(),
               dims.height(),
               dims.width());

    param.y->ZynqTensor()->flush();
    copy_properties(param);

    param.x->ZynqTensor()->invalidate();
    param.x->ZynqTensor()->flush();
  }
  std::string doc() const override { return "Copy IO from FPGA to HOST"; }
};

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyHostCHWToFpgaHWCCompute,
                     host_to_device)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyHostCHWToFpgaHWCCompute,
                     host_float_chw_to_device_fp16_hwc)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyFpgaToHostCHWCompute,
                     device_to_host_chw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(calib,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyFpgaToHostCHWCompute,
                     device_to_host_chw_calib)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyFpgaToHostCHWCompute,
                     device_to_host_hwc_chw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
